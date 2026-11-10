//! Enough of the NumPy `.npy` header to check rank, length, and that the file is intact.
//!
//! This matches `numpy.load(..., allow_pickle=False)` for the arrays physint writes:
//! a little-endian numeric dtype and a C-order shape.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub struct NpyHeader {
    pub shape: Vec<u64>,
    pub data_offset: u64,
    pub itemsize: u64,
}

#[derive(Debug)]
pub struct NpyError(pub String);

impl std::fmt::Display for NpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn read_header(path: &Path) -> Result<NpyHeader, NpyError> {
    let mut file = File::open(path).map_err(|err| NpyError(err.to_string()))?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|err| NpyError(err.to_string()))?;
    if &magic != b"\x93NUMPY" {
        return Err(NpyError("not a numpy file".into()));
    }
    let mut ver = [0u8; 2];
    file.read_exact(&mut ver)
        .map_err(|err| NpyError(err.to_string()))?;
    let header_len = match ver[0] {
        1 => {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf)
                .map_err(|err| NpyError(err.to_string()))?;
            u16::from_le_bytes(buf) as u64
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)
                .map_err(|err| NpyError(err.to_string()))?;
            u32::from_le_bytes(buf) as u64
        }
        other => return Err(NpyError(format!("unsupported npy version {other}"))),
    };
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| NpyError(err.to_string()))?;
    let text = String::from_utf8_lossy(&header);
    let shape = parse_shape(&text)?;
    let itemsize = parse_itemsize(&text)?;
    let data_offset = file
        .stream_position()
        .map_err(|err| NpyError(err.to_string()))?;
    let file_len = file
        .seek(SeekFrom::End(0))
        .map_err(|err| NpyError(err.to_string()))?;
    let n_items = shape
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim));
    let Some(n_items) = n_items else {
        return Err(NpyError("shape is too large".into()));
    };
    let Some(nbytes) = n_items.checked_mul(itemsize) else {
        return Err(NpyError("array is too large".into()));
    };
    if data_offset.saturating_add(nbytes) != file_len {
        return Err(NpyError(format!(
            "numpy data length does not match the header (offset {data_offset}, bytes {nbytes}, file {file_len})"
        )));
    }
    Ok(NpyHeader {
        shape,
        data_offset,
        itemsize,
    })
}

fn parse_shape(header: &str) -> Result<Vec<u64>, NpyError> {
    let key = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| NpyError("npy header has no shape".into()))?;
    let tail = &header[key..];
    let open = tail
        .find('(')
        .ok_or_else(|| NpyError("npy shape is not a tuple".into()))?;
    let close = tail[open..]
        .find(')')
        .ok_or_else(|| NpyError("npy shape is not a tuple".into()))?;
    let inside = &tail[open + 1..open + close];
    if inside.trim().is_empty() {
        return Ok(Vec::new());
    }
    inside
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map_err(|_| NpyError(format!("bad npy dimension {}", part.trim())))
        })
        .collect()
}

fn parse_itemsize(header: &str) -> Result<u64, NpyError> {
    let key = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| NpyError("npy header has no descr".into()))?;
    let tail = &header[key..];
    let quote = tail
        .find(['\'', '"'])
        .ok_or_else(|| NpyError("npy descr is not a string".into()))?;
    let rest = &tail[quote + 1..];
    let end = rest
        .find(['\'', '"'])
        .ok_or_else(|| NpyError("npy descr is not a string".into()))?;
    let descr = &rest[..end];
    // The first quoted string after 'descr' is the key's quote. Step to the value.
    let after_key = rest[end + 1..]
        .trim_start()
        .trim_start_matches(':')
        .trim_start();
    let value_quote = after_key
        .find(['\'', '"'])
        .ok_or_else(|| NpyError(format!("npy descr {descr} has no value")))?;
    let value_rest = &after_key[value_quote + 1..];
    let value_end = value_rest
        .find(['\'', '"'])
        .ok_or_else(|| NpyError("npy descr value is not closed".into()))?;
    let dtype = &value_rest[..value_end];
    let digits: String = dtype
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if digits.is_empty() {
        return Err(NpyError(format!("unsupported numpy dtype {dtype}")));
    }
    digits
        .parse::<u64>()
        .map_err(|_| NpyError(format!("unsupported numpy dtype {dtype}")))
}

pub fn shape_tuple(shape: &[u64]) -> String {
    match shape {
        [] => "()".to_string(),
        [one] => format!("({one},)"),
        many => {
            let parts: Vec<String> = many.iter().map(|dim| dim.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_version_1_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.npy");
        // 2x3 float32, header padded to the 64-byte alignment numpy uses.
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let mut header = dict.as_bytes().to_vec();
        header.push(b'\n');
        let prefix = 6 + 2 + 2;
        let pad = (16 - ((prefix + header.len()) % 16)) % 16;
        for _ in 0..pad {
            header.insert(header.len() - 1, b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 2 * 3 * 4]);
        std::fs::write(&path, bytes).unwrap();
        let parsed = read_header(&path).unwrap();
        assert_eq!(parsed.shape, vec![2, 3]);
        assert_eq!(parsed.itemsize, 4);
    }
}
