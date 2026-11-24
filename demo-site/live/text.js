// Any word, not only the pack's list: CLIP's tokenizer, ported from Hugging Face's CLIPTokenizer
// (byte-level BPE), for the text encoder that `ii export-text` adds to a words pack.
// tests/test_live_text.py checks it against the Python tokenizer.

const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
const MAX_TOKENS = 77;

// GPT-2's map from bytes to printable characters (bytes_to_unicode).
function byteChars() {
  const keep = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const chars = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) chars[b] = String.fromCharCode(keep.includes(b) ? b : 256 + extra++);
  return chars;
}

export class Tokenizer {
  // spec: { vocab: [token, ...] by id, merges: ["a b", ...] by rank, bos, eos }
  constructor(spec) {
    this.ids = new Map(spec.vocab.map((token, id) => [token, id]));
    this.ranks = new Map(spec.merges.map((pair, rank) => [pair, rank]));
    this.bos = spec.bos;
    this.eos = spec.eos;
    this.byte = byteChars();
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  bpe(token) {
    const hit = this.cache.get(token);
    if (hit) return hit;
    const chars = Array.from(token);
    let word = [...chars.slice(0, -1), chars[chars.length - 1] + "</w>"];
    if (word.length === 1) return [word[0]];
    for (;;) {
      let best = null;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = [word[i], word[i + 1]];
        }
      }
      if (!best) break;
      const [first, second] = best;
      const next = [];
      for (let i = 0; i < word.length; ) {
        const j = word.indexOf(first, i);
        if (j < 0) {
          next.push(...word.slice(i));
          break;
        }
        next.push(...word.slice(i, j));
        i = j;
        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          next.push(first + second);
          i += 2;
        } else {
          next.push(word[i]);
          i += 1;
        }
      }
      word = next;
      if (word.length === 1) break;
    }
    this.cache.set(token, word);
    return word;
  }

  // Token ids with the start and end tokens, cut to CLIP's 77.
  encode(text) {
    const clean = text.normalize("NFC").replace(/\s+/g, " ").trim().toLowerCase();
    const ids = [this.bos];
    for (const [piece] of clean.matchAll(PAT)) {
      const mapped = Array.from(this.utf8.encode(piece), (b) => this.byte[b]).join("");
      for (const part of this.bpe(mapped)) ids.push(this.ids.get(part) ?? this.eos);
    }
    if (ids.length > MAX_TOKENS - 1) ids.length = MAX_TOKENS - 1;
    ids.push(this.eos);
    return ids;
  }
}

// The prompt a word is read in, as the pack's word list was ("a photo of a {word}.").
export function promptFor(template, word) {
  const article = /^[aeiou]/i.test(word) ? "an" : "a";
  return template.replace("{article}", article).replace("{word}", word);
}
