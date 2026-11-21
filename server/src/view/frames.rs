//! Clip arrays to PNG frames.
//!
//! The arithmetic is float32 in the same order as NumPy 2 does it in the Python viewer, so the
//! pixels match: one or three channels are shown directly, any other channel count is a dark
//! field tinted by the strongest channel.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::npy;

/// A clip as float32, `[frames, height, width, channels]` in C order.
pub struct ClipArray {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub data: Vec<f32>,
    /// `True` when the file was `T×H×W` (no channel axis).
    pub rank3: bool,
}

const PALETTE: [[f32; 3]; 8] = [
    [230.0, 92.0, 48.0],
    [42.0, 128.0, 168.0],
    [46.0, 150.0, 96.0],
    [214.0, 168.0, 48.0],
    [132.0, 96.0, 186.0],
    [186.0, 72.0, 112.0],
    [96.0, 112.0, 124.0],
    [64.0, 140.0, 136.0],
];
const BACKGROUND: [f32; 3] = [18.0, 17.0, 15.0];

pub fn load_npy(path: &Path) -> Result<ClipArray, String> {
    let header = npy::read_header(path).map_err(|err| err.to_string())?;
    if header.fortran_order {
        return Err("Fortran-order arrays are not supported".into());
    }
    let shape: Vec<usize> = header.shape.iter().map(|dim| *dim as usize).collect();
    let (frames, height, width, channels, rank3) = match shape.as_slice() {
        [t, h, w] => (*t, *h, *w, 1, true),
        [t, h, w, c] => (*t, *h, *w, *c, false),
        _ => return Err(format!("expected a rank 3 or 4 array, got {}", npy::shape_tuple(&header.shape))),
    };
    let n = frames * height * width * channels;
    let mut file = std::fs::File::open(path).map_err(|err| err.to_string())?;
    file.seek(SeekFrom::Start(header.data_offset)).map_err(|err| err.to_string())?;
    let mut raw = vec![0u8; n * header.itemsize as usize];
    file.read_exact(&mut raw).map_err(|err| err.to_string())?;
    let data = decode(&header.descr, &raw, n)?;
    Ok(ClipArray {
        frames,
        height,
        width,
        channels,
        data,
        rank3,
    })
}

fn decode(descr: &str, raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let (order, kind) = match descr.as_bytes().first() {
        Some(b'<') | Some(b'|') | Some(b'=') => (Endian::Little, &descr[1..]),
        Some(b'>') => (Endian::Big, &descr[1..]),
        _ => (Endian::Little, descr),
    };
    macro_rules! read {
        ($ty:ty, $size:expr) => {{
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact($size) {
                let bytes: [u8; $size] = chunk.try_into().unwrap();
                let value = match order {
                    Endian::Little => <$ty>::from_le_bytes(bytes),
                    Endian::Big => <$ty>::from_be_bytes(bytes),
                };
                out.push(value as f32);
            }
            out
        }};
    }
    let data = match kind {
        "f4" => read!(f32, 4),
        "f8" => read!(f64, 8),
        "f2" => {
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = match order {
                    Endian::Little => u16::from_le_bytes([chunk[0], chunk[1]]),
                    Endian::Big => u16::from_be_bytes([chunk[0], chunk[1]]),
                };
                out.push(f16_to_f32(bits));
            }
            out
        }
        "u1" => raw.iter().map(|value| *value as f32).collect(),
        "i1" => raw.iter().map(|value| *value as i8 as f32).collect(),
        "b1" => raw.iter().map(|value| (*value != 0) as u8 as f32).collect(),
        "u2" => read!(u16, 2),
        "i2" => read!(i16, 2),
        "u4" => read!(u32, 4),
        "i4" => read!(i32, 4),
        "u8" => read!(u64, 8),
        "i8" => read!(i64, 8),
        other => return Err(format!("unsupported numpy dtype {other}")),
    };
    Ok(data)
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let value = if exp == 0 {
        (frac as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(((exp + 112) << 23) | (frac << 13))
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

impl ClipArray {
    /// The composite's full-scale value: 1 for images, else the clip's max.
    pub fn peak(&self) -> f64 {
        if self.rank3 || matches!(self.channels, 1 | 3 | 4) {
            return 1.0;
        }
        if self.data.is_empty() {
            return 1.0;
        }
        let mut peak = f32::NEG_INFINITY;
        for value in &self.data {
            if value.is_nan() {
                return 1.0;
            }
            peak = peak.max(*value);
        }
        let peak = peak as f64;
        if peak > 1e-8 {
            peak
        } else {
            1.0
        }
    }

    fn frame(&self, index: usize) -> &[f32] {
        let size = self.height * self.width * self.channels;
        &self.data[index * size..(index + 1) * size]
    }
}

/// `uint8` RGB of one frame, `height * width * 3`.
pub fn render_frame(clip: &ClipArray, index: usize, peak: f64) -> Vec<u8> {
    let raw = clip.frame(index);
    let pixels = clip.height * clip.width;
    let channels = clip.channels;
    let array: Vec<f32> = raw.iter().map(|value| nan_to_num(*value)).collect();
    let rgb: Vec<f32> = if channels == 1 {
        let tone = unit_interval(&array);
        tone.iter().flat_map(|value| [*value, *value, *value]).collect()
    } else if channels == 3 || channels == 4 {
        let mut first: Vec<f32> = Vec::with_capacity(pixels * 3);
        for pixel in 0..pixels {
            first.extend_from_slice(&array[pixel * channels..pixel * channels + 3]);
        }
        unit_interval(&first)
    } else {
        composite(&array, pixels, channels, peak)
    };
    rgb.iter().map(|value| to_u8(*value)).collect()
}

fn to_u8(value: f32) -> u8 {
    let clipped = clip01(value);
    let scaled = clipped * 255.0f32;
    scaled as u8
}

fn clip01(value: f32) -> f32 {
    // np.clip keeps NaN; casting NaN to uint8 is 0 here.
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

fn nan_to_num(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else if value == f32::INFINITY {
        f32::MAX
    } else if value == f32::NEG_INFINITY {
        f32::MIN
    } else {
        value
    }
}

fn unit_interval(array: &[f32]) -> Vec<f32> {
    let (mut peak, mut low) = (0.0f64, 0.0f64);
    if !array.is_empty() {
        let (mut hi, mut lo) = (f32::NEG_INFINITY, f32::INFINITY);
        for value in array {
            hi = hi.max(*value);
            lo = lo.min(*value);
        }
        peak = hi as f64;
        low = lo as f64;
    }
    if peak <= 1.5 && low >= -0.05 {
        return array.iter().map(|value| clip01(*value)).collect();
    }
    if peak <= 255.0 && low >= 0.0 {
        return array.iter().map(|value| clip01(*value / 255.0f32)).collect();
    }
    let span = if peak > low { peak - low } else { 1.0 };
    let (low32, span32) = (low as f32, span as f32);
    array.iter().map(|value| clip01((*value - low32) / span32)).collect()
}

fn composite(array: &[f32], pixels: usize, channels: usize, peak: f64) -> Vec<f32> {
    let scale = if peak > 1e-8 { peak } else { 1.0 };
    let floor = 0.05f64.max(0.05 * scale);
    let (floor32, denom32) = (floor as f32, (scale - floor) as f32);
    let mut out = Vec::with_capacity(pixels * 3);
    for pixel in 0..pixels {
        let values = &array[pixel * channels..(pixel + 1) * channels];
        let mut strength = values[0];
        let mut dominant = 0usize;
        for (index, value) in values.iter().enumerate().skip(1) {
            if *value > strength {
                strength = *value;
                dominant = index;
            }
        }
        let alpha = clip01((strength - floor32) / denom32);
        let color = PALETTE[dominant % PALETTE.len()];
        for channel in 0..3 {
            let mixed = BACKGROUND[channel] * (1.0f32 - alpha) + color[channel] * alpha;
            out.push(mixed / 255.0f32);
        }
    }
    out
}

/// Nearest-neighbour upscale so a small frame is at least 384 pixels on its long side.
pub fn upsample(rgb: &[u8], height: usize, width: usize) -> (Vec<u8>, usize, usize) {
    let longest = height.max(width).max(1);
    let scale = if longest >= 384 { 1 } else { 1.max(384usize.div_ceil(longest)) };
    if scale <= 1 {
        return (rgb.to_vec(), height, width);
    }
    let (out_h, out_w) = (height * scale, width * scale);
    let mut out = Vec::with_capacity(out_h * out_w * 3);
    for row in 0..out_h {
        let source = row / scale;
        for col in 0..out_w {
            let at = (source * width + col / scale) * 3;
            out.extend_from_slice(&rgb[at..at + 3]);
        }
    }
    (out, out_h, out_w)
}

/// An 8-bit RGB PNG.
pub fn encode_png(rgb: &[u8], height: usize, width: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(height * (width * 3 + 1));
    for row in 0..height {
        raw.push(0);
        raw.extend_from_slice(&rgb[row * width * 3..(row + 1) * width * 3]);
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    encoder.write_all(&raw).expect("in-memory write");
    let idat = encoder.finish().expect("in-memory write");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &idat);
    chunk(&mut png, b"IEND", &[]);
    png
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(tag);
    hasher.update(data);
    out.extend_from_slice(&hasher.finalize().to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_multichannel_frame_tints_the_hot_patch() {
        let clip = ClipArray {
            frames: 1,
            height: 1,
            width: 2,
            channels: 8,
            data: {
                let mut data = vec![0.0f32; 16];
                data[8 + 2] = 1.0;
                data
            },
            rank3: false,
        };
        let rgb = render_frame(&clip, 0, clip.peak());
        assert_eq!(&rgb[0..3], &[18, 17, 15]);
        // Channel 2's colour, give or take the float32 round trip through 1/255.
        for (got, want) in rgb[3..6].iter().zip([46u8, 150, 96]) {
            assert!(got.abs_diff(want) <= 1, "{got} vs {want}");
        }
    }

    #[test]
    fn png_has_the_signature_and_three_chunks() {
        let png = encode_png(&[255, 0, 0], 1, 1);
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(png.ends_with(&[0xAE, 0x42, 0x60, 0x82]));
    }
}
