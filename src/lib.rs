use std::io::{self, Read, Write};
use std::mem::size_of;

#[derive(Debug, Default)]
pub struct BitWriter {
    buffer: Vec<u8>,
    scratch: u64,
    scratch_bits: usize,
}

impl BitWriter {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            scratch: 0,
            scratch_bits: 0,
        }
    }

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

    pub fn write_bool(&mut self, value: bool) -> Result<(), io::Error> {
        self.write_bits(value as u32, 1)
    }

    pub fn write_u8(&mut self, byte: u8) -> Result<(), io::Error> {
        self.write_bits(byte as u32, 8)
    }

    pub fn write_u16(&mut self, value: u16) -> Result<(), io::Error> {
        self.write_bits(value as u32, 16)
    }

    pub fn write_u32(&mut self, value: u32) -> Result<(), io::Error> {
        self.write_bits(value, 32)
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), io::Error> {
        let low_bits = value as u32;
        let high_bits = (value >> 32) as u32;
        self.write_bits(low_bits, 32)?;
        self.write_bits(high_bits, 32)
    }

    pub fn write_i16(&mut self, value: i16) -> Result<(), io::Error> {
        self.write_bits(value as u32, 16)
    }

    pub fn write_i32(&mut self, value: i32) -> Result<(), io::Error> {
        self.write_bits(value as u32, 32)
    }

    pub fn write_i64(&mut self, value: i64) -> Result<(), io::Error> {
        self.write_u64(value as u64)
    }

    pub fn write_varint_u16(&mut self, value: u16) -> Result<(), io::Error> {
        self.write_varint_u64(value as u64)
    }

    pub fn write_varint_u32(&mut self, value: u32) -> Result<(), io::Error> {
        self.write_varint_u64(value as u64)
    }

    // Split the value in 7 bit chunks, the 8 bit is used as a flag indicate if there are more bytes
    // to read. The first 8 bytes are flag + 7 bits of value. If we get to the last byte, there is
    // no need for the flag and we can use the whole byte as value. We need at max 9 bytes to store the u64.
    pub fn write_varint_u64(&mut self, mut value: u64) -> Result<(), io::Error> {
        for _ in 0..8 {
            let mut t = value as u8;
            // Get 7 bits on information
            t &= 0b011111111u8;
            value >>= 7;

            // Use last bit of the byte as a flag
            let more_to_write = value != 0;
            if more_to_write {
                t |= 0b10000000u8;
            }

            self.write_u8(t)?;

            if !more_to_write {
                return Ok(());
            }
        }

        // If we got here, it means we have bits at the highest byte
        // We use the full 8 bits at the last step
        self.write_u8(value as u8)
    }

    pub fn write_varint_i16(&mut self, value: i16) -> Result<(), io::Error> {
        let value = zig_zag_encode(value as i64);
        self.write_varint_u64(value)
    }

    pub fn write_varint_i32(&mut self, value: i32) -> Result<(), io::Error> {
        let value = zig_zag_encode(value as i64);
        self.write_varint_u64(value)
    }

    pub fn write_varint_i64(&mut self, value: i64) -> Result<(), io::Error> {
        let value = zig_zag_encode(value);
        self.write_varint_u64(value)
    }

    pub fn write_f32(&mut self, value: f32) -> Result<(), io::Error> {
        self.write_u32(value.to_bits())
    }

    pub fn write_f64(&mut self, value: f64) -> Result<(), io::Error> {
        self.write_u64(value.to_bits())
    }
}

impl Write for BitWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Make sure the buffer is aligned
        self.align()?;

        // Get remaining bytes to fill scratch
        let mut head_bytes = ((32 - self.scratch_bits) / 8) % 4;

        if head_bytes > buf.len() {
            // Not enough bytes to fill it
            head_bytes = buf.len();
        }

        for &value in buf.iter().take(head_bytes) {
            self.write_bits(value as u32, 8)?;
        }

        if head_bytes == buf.len() {
            return Ok(buf.len());
        }

        // The scratch is filled, let's flush it so we have a clean scratch
        self.flush_bits()?;
        assert_eq!(self.align_bits(), 0);

        // Now that the scratch is empty, we can copy the bytes directly to the buffer
        const U32_SIZE: usize = size_of::<u32>();
        let num_words = (buf.len() - head_bytes) / U32_SIZE;
        if num_words > 0 {
            self.buffer
                .extend_from_slice(&buf[head_bytes..head_bytes + num_words * U32_SIZE]);
        }

        // The buffer might have some small part at the end that is less then 4 bytes
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
    pub fn new(buffer: &'a [u8]) -> Result<Self, io::Error> {
        if buffer.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "BitReader buffer must have the length as a multiple of 4",
            ));
        }
        Ok(Self {
            buffer,
            scratch: 0,
            scratch_bits: 0,
            bits_read: 0,
        })
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
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid padding, alignment bits must all be 0",
                ));
            }
        }

        Ok(())
    }

    pub fn read_bool(&mut self) -> Result<bool, io::Error> {
        Ok(self.read_bits(1)? == 1)
    }

    pub fn read_u8(&mut self) -> Result<u8, io::Error> {
        Ok(self.read_bits(8)? as u8)
    }

    pub fn read_u16(&mut self) -> Result<u16, io::Error> {
        Ok(self.read_bits(16)? as u16)
    }

    pub fn read_u32(&mut self) -> Result<u32, io::Error> {
        self.read_bits(32)
    }

    pub fn read_u64(&mut self) -> Result<u64, io::Error> {
        let low_bits = self.read_bits(32)?;
        let high_bits = self.read_bits(32)?;

        let value = low_bits as u64 | ((high_bits as u64) << 32);
        Ok(value)
    }

    pub fn read_i16(&mut self) -> Result<i16, io::Error> {
        Ok(self.read_bits(16)? as i16)
    }

    pub fn read_i32(&mut self) -> Result<i32, io::Error> {
        Ok(self.read_bits(32)? as i32)
    }

    pub fn read_i64(&mut self) -> Result<i64, io::Error> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_varint_u16(&mut self) -> Result<u16, io::Error> {
        let value = self.read_varint_u64()?;
        Ok(value as u16)
    }

    pub fn read_varint_u32(&mut self) -> Result<u32, io::Error> {
        let value = self.read_varint_u64()?;
        Ok(value as u32)
    }

    pub fn read_varint_u64(&mut self) -> Result<u64, io::Error> {
        let mut result: u64 = 0;
        for i in 0..8 {
            let byte = self.read_u8()?;
            // Retrieve the flag stored in the last bit
            let stop_reading = (byte & 0b10000000u8) == 0;

            // Retrieve 7 bits of information and store them in the result
            let value = (byte & 0b01111111u8) as u64;
            result |= value << (i * 7);

            if stop_reading {
                return Ok(result);
            }
        }

        // If we got here, it means we have bits at the highest byte
        // We use the full 8 bits at the last step
        let value = self.read_u8()? as u64;
        result |= value << 56;

        Ok(result)
    }

    pub fn read_varint_i16(&mut self) -> Result<i16, io::Error> {
        let value = self.read_varint_u64()?;
        let value = zig_zag_decode(value);
        Ok(value as i16)
    }

    pub fn read_varint_i32(&mut self) -> Result<i32, io::Error> {
        let value = self.read_varint_u64()?;
        let value = zig_zag_decode(value);
        Ok(value as i32)
    }

    pub fn read_varint_i64(&mut self) -> Result<i64, io::Error> {
        let value = self.read_varint_u64()?;
        let value = zig_zag_decode(value);
        Ok(value)
    }

    pub fn read_f32(&mut self) -> Result<f32, io::Error> {
        let value = self.read_u32()?;
        Ok(f32::from_bits(value))
    }

    pub fn read_f64(&mut self) -> Result<f64, io::Error> {
        let value = self.read_u64()?;
        Ok(f64::from_bits(value))
    }
}

impl<'a> Read for BitReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Make sure the buffer is aligned
        self.align()?;

        // Get number of bytes to read from the scratch
        let mut head_bytes = (self.scratch_bits / 8) % 4;

        // Not enough bytes to read all the scratch
        if head_bytes > buf.len() {
            head_bytes = buf.len();
        }

        for value in buf.iter_mut().take(head_bytes) {
            *value = self.read_bits(8)? as u8;
        }

        if head_bytes == buf.len() {
            return Ok(buf.len());
        }

        // Now that the scratch is empty we can read directly from the buffer
        const U32_SIZE: usize = size_of::<u32>();
        let num_words = (buf.len() - head_bytes) / U32_SIZE;
        if num_words > 0 {
            self.buffer
                .read_exact(&mut buf[head_bytes..head_bytes + (num_words * U32_SIZE)])?;
            self.bits_read += num_words * 32;
        }

        // If there is less than 4 bytes remaining, we need to add them to the scratch and read
        let tail_start = head_bytes + num_words * U32_SIZE;
        let tail_bytes = buf.len() - tail_start;
        assert!(tail_bytes < 4);

        for i in 0..tail_bytes {
            buf[tail_start + i] = self.read_bits(8)? as u8;
        }

        Ok(buf.len())
    }
}

/// Convert a signed number to an unsigned number with zig-zag encoding
/// 0, -1, +1, -2, +2 ... becomes 0, 1, 2, 3, 4 ...
/// We use this for better varint encoding
#[inline(always)]
fn zig_zag_encode(value: i64) -> u64 {
    if value < 0 {
        !(value as u64) * 2 + 1
    } else {
        (value as u64) * 2
    }
}

/// Convert an unsigned number to as signed number with zig-zag encoding.
/// 0, 1, 2, 3, 4 ... becomes 0, -1, +1, -2, +2 ...
/// We use this for better varint encoding
#[inline(always)]
fn zig_zag_decode(value: u64) -> i64 {
    if value % 2 == 0 {
        (value / 2) as i64
    } else {
        !(value / 2) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[test]
    fn usage() {
        let mut writer = BitWriter::default();

        // You can write bools and they use only one bit
        writer.write_bool(true).unwrap();

        // You can write values with how many bits you wish
        // Write the value 3 with only 2 bits
        let value: u32 = 3;
        writer.write_bits(value, 2).unwrap();

        // You can also write write basic types
        writer.write_u8(0).unwrap(); // uses 8 bits
        writer.write_u16(1).unwrap(); // uses 16 bits
        writer.write_u32(2).unwrap(); // uses 32 bits
        writer.write_u64(3).unwrap(); // uses 64 bits
        writer.write_i64(-1).unwrap(); // uses 16 bits
        writer.write_i32(-2).unwrap(); // uses 32 bits
        writer.write_i64(-3).unwrap(); // uses 64 bits

        // But you can also use the varint encoding variation
        // That tries to use the least number of bits to encode the value
        writer.write_varint_u16(1).unwrap(); // uses 8 bits
        writer.write_varint_u32(2).unwrap(); // uses 8 bits
        writer.write_varint_u64(3).unwrap(); // uses 8 bits
        writer.write_varint_i16(-1).unwrap(); // uses 8 bits
        writer.write_varint_i32(-2).unwrap(); // uses 8 bits
        writer.write_varint_i64(-3).unwrap(); // uses 8 bits

        // Bigger values will use more than one byte

        // Float types
        writer.write_f32(1.0).unwrap(); // uses 32 bits
        writer.write_f64(2.0).unwrap(); // uses 64 bits

        // Since the serializers impl Write/Read, we can use bincode
        // Or write/read bytes directly to/from them
        let bytes = vec![7u8; 20];
        writer.write_all(&bytes).unwrap();

        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
        struct SimpleStruct {
            value: u64,
            string: String,
            array: [u16; 12],
        }

        let message = SimpleStruct {
            value: 999999999999,
            string: "some text to serialize".to_owned(),
            array: [500; 12],
        };

        // Serialize using bincode passing a writer
        bincode::serialize_into(&mut writer, &message).unwrap();

        // Consume the writer to get the buffer, so we can create the reader
        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        // Now to the reading, just replace write for read, and do it in the same order :)
        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_bits(2).unwrap(), 3);

        assert_eq!(reader.read_u8().unwrap(), 0);
        assert_eq!(reader.read_u16().unwrap(), 1);
        assert_eq!(reader.read_u32().unwrap(), 2);
        assert_eq!(reader.read_u64().unwrap(), 3);
        assert_eq!(reader.read_i64().unwrap(), -1);
        assert_eq!(reader.read_i32().unwrap(), -2);
        assert_eq!(reader.read_i64().unwrap(), -3);

        assert_eq!(reader.read_varint_u16().unwrap(), 1);
        assert_eq!(reader.read_varint_u32().unwrap(), 2);
        assert_eq!(reader.read_varint_u64().unwrap(), 3);
        assert_eq!(reader.read_varint_i16().unwrap(), -1);
        assert_eq!(reader.read_varint_i32().unwrap(), -2);
        assert_eq!(reader.read_varint_i64().unwrap(), -3);

        assert_eq!(reader.read_f32().unwrap(), 1.0);
        assert_eq!(reader.read_f64().unwrap(), 2.0);

        let mut new_bytes = vec![0u8; bytes.len()];
        reader.read_exact(&mut new_bytes).unwrap();
        assert_eq!(bytes, new_bytes);

        let de_message: SimpleStruct = bincode::deserialize_from(&mut reader).unwrap();
        assert_eq!(message, de_message);
    }

    #[test]
    fn bit_writer_reader_test() {
        let mut writer = BitWriter::default();

        writer.write_bits(3, 2).unwrap();
        writer.write_bits(5, 5).unwrap();
        // Now it's not aligned

        let bytes = vec![0, 1, 2, 3, 4, 5, 6, 7];
        writer.write_all(&bytes).unwrap();

        writer.write_bits(7, 12).unwrap();
        writer.write_bits(1, 1).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert_eq!(reader.read_bits(2).unwrap(), 3);
        assert_eq!(reader.read_bits(5).unwrap(), 5);
        let mut new_bytes = vec![0u8; bytes.len()];
        reader.read_exact(&mut new_bytes).unwrap();
        assert_eq!(new_bytes, bytes);
        assert_eq!(reader.read_bits(12).unwrap(), 7);
        assert_eq!(reader.read_bits(1).unwrap(), 1);
    }

    #[test]
    fn bit_read_write_aligned() {
        let mut writer = BitWriter::default();

        let bytes = vec![0, 1, 2, 3, 4, 5, 6, 7];
        writer.write_all(&bytes).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        let mut new_bytes = vec![0u8; bytes.len()];
        reader.read_exact(&mut new_bytes).unwrap();
        assert_eq!(new_bytes, bytes);
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestMessage {
        value: u64,
        array: [u16; 12],
        string: String,
    }

    #[test]
    fn bincode_aligned() {
        let mut writer = BitWriter::default();

        let message = TestMessage {
            value: 999999999999,
            array: [500; 12],
            string: "just a test string".to_owned(),
        };

        bincode::serialize_into(&mut writer, &message).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        let de_message: TestMessage = bincode::deserialize_from(&mut reader).unwrap();

        assert_eq!(message, de_message);
    }

    #[test]
    fn bincode_not_aligned() {
        let mut writer = BitWriter::default();

        let message = TestMessage {
            value: 999999999999,
            array: [500; 12],
            string: "just a test string".to_owned(),
        };

        writer.write_bits(3, 5).unwrap();

        bincode::serialize_into(&mut writer, &message).unwrap();

        writer.write_bits(1, 2).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert_eq!(reader.read_bits(5).unwrap(), 3);
        let de_message: TestMessage = bincode::deserialize_from(&mut reader).unwrap();
        assert_eq!(reader.read_bits(2).unwrap(), 1);

        assert_eq!(message, de_message);
    }

    #[test]
    fn varint_aligned() {
        let mut writer = BitWriter::default();

        writer.write_varint_u64(5).unwrap();
        assert_eq!(writer.bits_written(), 8);

        let high_number = 0xffa0000000000000u64;
        writer.write_varint_u64(high_number).unwrap();
        assert_eq!(writer.bits_written(), 8 + (9 * 8));

        writer.write_varint_u32(320000).unwrap();
        writer.write_varint_u16(16000).unwrap();

        let high_negative_number = -0xffa000000000000i64;
        writer.write_varint_i64(high_negative_number).unwrap();
        writer.write_varint_i32(-320000).unwrap();
        writer.write_varint_i16(-16000).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert_eq!(reader.read_varint_u64().unwrap(), 5);
        assert_eq!(reader.read_varint_u64().unwrap(), high_number);
        assert_eq!(reader.read_varint_u32().unwrap(), 320000);
        assert_eq!(reader.read_varint_u16().unwrap(), 16000);

        assert_eq!(reader.read_varint_i64().unwrap(), high_negative_number);
        assert_eq!(reader.read_varint_i32().unwrap(), -320000);
        assert_eq!(reader.read_varint_i16().unwrap(), -16000);
    }

    #[test]
    fn varint_not_aligned() {
        let mut writer = BitWriter::default();

        writer.write_bits(3, 5).unwrap();

        writer.write_varint_u64(5).unwrap();

        let high_number = 0xffa0000000000000u64;
        writer.write_varint_u64(high_number).unwrap();

        writer.write_varint_u32(320000).unwrap();
        writer.write_varint_u16(16000).unwrap();

        let high_negative_number = -0xffa000000000000i64;
        writer.write_varint_i64(high_negative_number).unwrap();
        writer.write_varint_i32(-320000).unwrap();
        writer.write_varint_i16(-16000).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert_eq!(reader.read_bits(5).unwrap(), 3);

        assert_eq!(reader.read_varint_u64().unwrap(), 5);
        assert_eq!(reader.read_varint_u64().unwrap(), high_number);
        assert_eq!(reader.read_varint_u32().unwrap(), 320000);
        assert_eq!(reader.read_varint_u16().unwrap(), 16000);

        assert_eq!(reader.read_varint_i64().unwrap(), high_negative_number);
        assert_eq!(reader.read_varint_i32().unwrap(), -320000);
        assert_eq!(reader.read_varint_i16().unwrap(), -16000);
    }

    #[test]
    fn bool() {
        let mut writer = BitWriter::default();
        writer.write_bool(true).unwrap();
        writer.write_bool(false).unwrap();
        writer.write_bool(true).unwrap();
        writer.write_bool(true).unwrap();
        writer.write_bool(false).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert!(reader.read_bool().unwrap());
        assert!(!reader.read_bool().unwrap());
        assert!(reader.read_bool().unwrap());
        assert!(reader.read_bool().unwrap());
        assert!(!reader.read_bool().unwrap());
    }

    #[test]
    fn float() {
        let mut writer = BitWriter::default();
        writer.write_bool(true).unwrap();

        writer.write_f32(1234.5678).unwrap();
        writer.write_f64(12345.6789).unwrap();

        let writer_bytes = writer.consume().unwrap();
        let mut reader = BitReader::new(&writer_bytes).unwrap();

        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_f32().unwrap(), 1234.5678);
        assert_eq!(reader.read_f64().unwrap(), 12345.6789);
    }
}
