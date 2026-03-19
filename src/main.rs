use std::{
    error::Error,
    fs::File,
    io::{Error as IoError, ErrorKind},
};

fn read_data_set() -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let file = File::open("training_data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut records: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        if record.len() != 101 {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidData,
                "The training_data is not 101 data points long.",
            )));
        }

        let row: Vec<f64> = record
            .iter()
            .map(|field| field.parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;
        records.push(row);
    }

    let (training_data, evaluation_data) = records.split_at((records.len() as f64 * 0.8) as usize);

    Ok((training_data.to_vec(), evaluation_data.to_vec()))
}

fn main() {
    let (training_data, evaluation_data) = read_data_set().unwrap();
    println!("{:?}", training_data);
}
