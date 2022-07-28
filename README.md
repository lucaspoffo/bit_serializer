# Bit Serializer
[![Latest version](https://img.shields.io/crates/v/bit_serializer.svg)](https://crates.io/crates/bit_serializer)
[![Documentation](https://docs.rs/bit_serializer/badge.svg)](https://docs.rs/bit_serializer)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Serialize data with control of every bit. Contains auxiliary methods for most basic types and
varint encoding for integer types.

### Usage

```rust
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
```

