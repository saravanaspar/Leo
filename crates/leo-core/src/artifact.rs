//! Strong, dependency-free artifact digests used for identity and integrity.
//!
//! Leo uses SHA-256 for durable artifact identity.  The implementation lives in
//! `leo-core` so dataset, checkpoint, resume, and synchronization code all use
//! one canonical digest type and algorithm.

use crate::{LeoError, LeoResult};
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub const DIGEST_ALGORITHM: &str = "sha256";
pub const DIGEST_BYTES: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ArtifactDigest([u8; DIGEST_BYTES]);

impl ArtifactDigest {
    pub const ZERO: Self = Self([0; DIGEST_BYTES]);

    pub const fn from_array(bytes: [u8; DIGEST_BYTES]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; DIGEST_BYTES] {
        &self.0
    }

    pub const fn is_zero(&self) -> bool {
        let mut index = 0usize;
        while index < DIGEST_BYTES {
            if self.0[index] != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(DIGEST_BYTES * 2);
        for byte in self.0 {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    pub fn from_hex(value: &str) -> LeoResult<Self> {
        if value.len() != DIGEST_BYTES * 2 {
            return Err(LeoError::internal(format!(
                "invalid SHA-256 digest length: {}",
                value.len()
            )));
        }
        let mut bytes = [0u8; DIGEST_BYTES];
        let raw = value.as_bytes();
        for index in 0..DIGEST_BYTES {
            bytes[index] = (hex_nibble(raw[index * 2])? << 4) | hex_nibble(raw[index * 2 + 1])?;
        }
        Ok(Self(bytes))
    }
}

impl Display for ArtifactDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl Debug for ArtifactDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ArtifactDigest({})", self.to_hex())
    }
}

fn hex_nibble(byte: u8) -> LeoResult<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(LeoError::internal("invalid hexadecimal SHA-256 digest")),
    }
}

#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.saturating_add(input.len() as u64);

        if self.buffer_len != 0 {
            let needed = 64 - self.buffer_len;
            let take = needed.min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&input[..take]);
            self.buffer_len += take;
            input = &input[take..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            }
        }

        while input.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[..64]);
            self.compress(&block);
            input = &input[64..];
        }

        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    pub fn finalize(mut self) -> ArtifactDigest {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffer_len != 56 {
            self.update(&[0]);
        }
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buffer_len, 0);

        let mut digest = [0u8; DIGEST_BYTES];
        for (index, word) in self.state.iter().enumerate() {
            digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        ArtifactDigest::from_array(digest)
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut schedule = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes(chunk.try_into().expect("four-byte SHA word"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];

        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

pub fn digest_bytes(bytes: &[u8]) -> ArtifactDigest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize()
}

pub fn digest_reader(mut reader: impl Read) -> std::io::Result<ArtifactDigest> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize())
}

pub fn digest_file(path: impl AsRef<Path>) -> std::io::Result<ArtifactDigest> {
    digest_reader(File::open(path)?)
}

pub fn write_digest_hex(mut writer: impl Write, digest: ArtifactDigest) -> std::io::Result<()> {
    writer.write_all(digest.to_hex().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{digest_bytes, ArtifactDigest};

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            digest_bytes(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest_bytes(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn digest_hex_round_trip() {
        let digest = digest_bytes(b"Leo v1.0.0");
        assert_eq!(ArtifactDigest::from_hex(&digest.to_hex()).unwrap(), digest);
    }
}
