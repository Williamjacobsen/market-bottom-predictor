use std::{error::Error, fs::File};

#[derive(Debug)]
struct Row {
    ticker: String,
    _index: usize,
    price: f64,
    is_minima: bool,
}

fn get_data() -> Result<(), Box<dyn Error>> {
    let file = File::open("training_data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    for result in rdr.records() {
        let record = result?;

        let row = Row {
            ticker: record[0].to_string(),
            _index: record[1].parse::<usize>()?,
            price: record[2].parse::<f64>()?,
            is_minima: record[3].parse::<u8>()? == 1,
        };
        println!("{:?}", row);
    }

    Ok(())
}

fn main() {
    get_data().unwrap();
}
