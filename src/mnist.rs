extern crate nalgebra as na;
extern crate flate2;
extern crate byteorder;

use std::io::{Read, Cursor};
use std::fs::File;
use std::mem::size_of;

use self::na::DMatrix;
use self::flate2::read::GzDecoder;
use self::byteorder::{BigEndian, ReadBytesExt};

pub fn import_data(path: &'static str) -> Vec<u8> {
    let mut content:Vec<u8> = Vec::new();
    let file = File::open(path)
        .expect(&format!("Failed to open file: {}", path)[..]);
    GzDecoder::new(file).read_to_end(&mut content).unwrap();
    return content
}

pub fn get_labels(data: &[u8]) -> Result<&[u8], &str> {
    let mut rdr = Cursor::new(data);
    let magic = rdr.read_u32::<BigEndian>();
    let count = rdr.read_u32::<BigEndian>();
    match (magic, count) {
        (Ok(magic), Ok(count)) => {
            if magic == 0x00000801 {
                if count as usize == (data.len() -  size_of::<u32>() * 2) {
                    Ok(&data[size_of::<u32>() * 2..])
                } else {
                    Err("invalid sizes")
                }
            } else {
                Err("invalid magic")
            }
        },
        _ => Err("could not read header")
    }
}

#[derive(Debug)]
pub struct Images<'a> {
    size: (usize, usize),
    data: &'a[u8],
}

impl<'a> Images<'a> {
    pub fn new<'b>(data: &'b [u8]) -> Result<Images<'b>, &str> {
        let mut rdr = Cursor::new(data);
        let magic = rdr.read_u32::<BigEndian>();
        let count = rdr.read_u32::<BigEndian>();
        let rows = rdr.read_u32::<BigEndian>();
        let cols = rdr.read_u32::<BigEndian>();
        match (magic, count, rows, cols) {
            (Ok(magic), Ok(count), Ok(rows), Ok(cols)) => {
                if magic == 0x00000803 {
                    if (count * rows * cols) as usize == (data.len() -  size_of::<u32>() * 4) {
                        Ok(Images { size: (rows as usize, cols as usize), data: &data[size_of::<u32>() * 4..] })
                    } else {
                        Err("invalid sizes")
                    }
                } else {
                    Err("invalid magic")
                }
            },
            _ => Err("could not read header")
        }
    }

    pub fn get<'b>(&self, n: usize) -> Option<DMatrix<u8>> {
        if n + self.size.1 <= self.data.len() {
            Some(DMatrix::<u8>::from_iterator(self.size.0, self.size.1,
                self.data.iter()
                    .skip(self.size.0 * self.size.1 * n)
                    .take(self.size.0 * self.size.1)
                    .map(|x| *x)))
        } else {
            None
        }
    }

    pub fn get_flat<'b>(&self, n: usize) -> Option<DMatrix<u8>> {
        if n + self.size.1 <= self.data.len() {
            Some(DMatrix::<u8>::from_iterator(self.size.0 * self.size.1, 1,
                self.data.iter()
                    .skip(self.size.0 * self.size.1 * n)
                    .take(self.size.0 * self.size.1)
                    .map(|x| *x)))
        } else {
            None
        }
    }
}
