use std::io::{self, Read, Write};

#[derive(Debug, Default)]
pub struct BitWriter {
    buffer: Vec<u8>,
    scratch: u64,
    scratch_bits: usize,
}

impl BitWriter {
    pub fn consume(mut self) -> Result<Vec<u8>, io::Error> {
        self.flush_bits()?;
        Ok(self.buffer)
    }

    pub fn write_bits(&mut self, value: u32, bits: usize) -> Result<(), io::Error> {
        assert!(bits <= 32);

        self.scratch |= (value as u64) << self.scratch_bits;
        self.scratch_bits += bits;

        if self.scratch_bits >= 32 {
            let bytes = (self.scratch as u32).to_le_bytes();
            self.buffer.write_all(&bytes)?;
            self.scratch >>= 32;
            self.scratch_bits -= 32;
        }

        Ok(())
    }

    pub fn align(&mut self) -> Result<(), io::Error> {
        let remainder_bits = self.scratch_bits % 8;
        if remainder_bits != 0 {
            self.write_bits(0, 8 - remainder_bits)?;
            assert!(self.scratch_bits % 8 == 0);
        }
        Ok(())
    }

    pub fn flush_bits(&mut self) -> Result<(), io::Error> {
        if self.scratch_bits != 0 {
            let bytes = (self.scratch as u32).to_le_bytes();
            self.buffer.write_all(&bytes)?;
            self.scratch = 0;
            self.scratch_bits = 0;
        }
        Ok(())
    }

    pub fn bits_written(&self) -> usize {
        self.buffer.len() * 8 + self.scratch_bits
    }

    fn align_bits(&self) -> usize {
        (8 - (self.scratch_bits % 8)) % 8
    }
}

impl Write for BitWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Make sure the buffer is aligned
        self.align()?;
        let mut head_bytes = (4 - ((self.bits_written() % 32) / 8)) % 4;
        if head_bytes > buf.len() {
            head_bytes = buf.len();
        }

        for &value in buf.iter().take(head_bytes) {
            self.write_bits(value as u32, 8)?;
        }

        if head_bytes == buf.len() {
            return Ok(buf.len());
        }

        self.flush_bits()?;
        assert_eq!(self.align_bits(), 0);

        const U32_SIZE: usize = std::mem::size_of::<u32>();
        let num_words = (buf.len() - head_bytes) / U32_SIZE;
        if num_words > 0 {
            self.buffer
                .extend_from_slice(&buf[head_bytes..head_bytes + num_words * U32_SIZE]);
        }

        let tail_start = head_bytes + num_words * U32_SIZE;
        let tail_bytes = buf.len() - tail_start;
        assert!(tail_bytes < 4);

        for i in 0..tail_bytes {
            self.write_bits(buf[tail_start + i] as u32, 8)?;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_bits()
    }
}

#[derive(Debug)]
pub struct BitReader<'a> {
    buffer: &'a [u8],
    scratch: u64,
    scratch_bits: usize,
    bits_read: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            scratch: 0,
            scratch_bits: 0,
            bits_read: 0,
        }
    }

    pub fn read_bits(&mut self, bits: usize) -> Result<u32, io::Error> {
        assert!(bits <= 32);

        if self.scratch_bits < bits {
            let mut word = [0u8; 4];
            self.buffer.read_exact(&mut word)?;
            let word = u32::from_le_bytes(word);
            self.scratch |= (word as u64) << self.scratch_bits;
            self.scratch_bits += 32;
        }

        assert!(self.scratch_bits >= bits);

        let output = (self.scratch & ((1u64 << bits) - 1)) as u32;
        self.scratch >>= bits;
        self.scratch_bits -= bits;
        self.bits_read += bits;

        Ok(output)
    }

    pub fn align(&mut self) -> Result<(), io::Error> {
        let remainder_bits = self.bits_read % 8;
        if remainder_bits != 0 {
            let value = self.read_bits(8 - remainder_bits)?;
            assert_eq!(self.bits_read % 8, 0);
            // When aligning, the padded value should be 0,
            // if the returned value is different, something went wrong
            if value != 0 {
                return Err(io::ErrorKind::InvalidData.into());
            }
        }

        Ok(())
    }
}

impl<'a> Read for BitReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Make sure the buffer is aligned
        self.align()?;

        let mut head_bytes = (4 - ((self.bits_read % 32) / 8)) % 4;
        if head_bytes > buf.len() {
            head_bytes = buf.len();
        }
        for value in buf.iter_mut().take(head_bytes) {
            *value = self.read_bits(8)? as u8;
        }
        if head_bytes == buf.len() {
            return Ok(buf.len());
        }

        const U32_SIZE: usize = std::mem::size_of::<u32>();
        let num_words = (buf.len() - head_bytes) / U32_SIZE;
        if num_words > 0 {
            self.buffer
                .read_exact(&mut buf[head_bytes..head_bytes + (num_words * U32_SIZE)])?;
            self.bits_read += num_words * 32;
        }

        let tail_start = head_bytes + num_words * U32_SIZE;
        let tail_bytes = buf.len() - tail_start;
        assert!(tail_bytes < 4);

        for i in 0..tail_bytes {
            buf[tail_start + i] = self.read_bits(8)? as u8;
        }

        Ok(buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_reader_test() {
        let mut writer = BitWriter::default();

        writer.write_bits(3, 2).unwrap();
        writer.write_bits(5, 5).unwrap();
        // Now it's not aligned

        let bytes = vec![0, 1, 2, 3, 4, 5, 6, 7];
        writer.write(&bytes).unwrap();

        writer.write_bits(7, 12).unwrap();
        writer.write_bits(1, 1).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes);

        assert_eq!(reader.read_bits(2).unwrap(), 3);
        assert_eq!(reader.read_bits(5).unwrap(), 5);
        let mut new_bytes = vec![0u8; bytes.len()];
        reader.read(&mut new_bytes).unwrap();
        assert_eq!(new_bytes, bytes);
        assert_eq!(reader.read_bits(12).unwrap(), 7);
        assert_eq!(reader.read_bits(1).unwrap(), 1);
    }
}
