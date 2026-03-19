use std::{
    error::Error,
    fs::File,
    io::{Error as IoError, ErrorKind},
};

fn min_max_normalization(input: &mut Vec<f64>) {
    let min = input.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let max = input.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    for x in input.iter_mut() {
        *x = (*x - min) / (max - min);
    }
}

fn get_data() -> Result<((Vec<Vec<f64>>, Vec<f64>), (Vec<Vec<f64>>, Vec<f64>)), Box<dyn Error>> {
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

    let split_idx = (records.len() as f64 * 0.8) as usize;
    let (training_records, evaluation_records) = records.split_at(split_idx);

    let (training_inputs, training_labels): (Vec<Vec<f64>>, Vec<f64>) = training_records
        .iter()
        .map(|record| {
            let (input, label) = record.split_at(100);
            let mut input = input.to_vec();
            min_max_normalization(&mut input);
            (input.to_vec(), label[0])
        })
        .unzip();

    let (evaluation_inputs, evaluation_labels): (Vec<Vec<f64>>, Vec<f64>) = evaluation_records
        .iter()
        .map(|record| {
            let (input, label) = record.split_at(100);
            let mut input = input.to_vec();
            min_max_normalization(&mut input);
            (input.to_vec(), label[0])
        })
        .unzip();

    Ok((
        (training_inputs, training_labels),
        (evaluation_inputs, evaluation_labels),
    ))
}

fn main() {
    let ((training_inputs, training_labels), (evaluation_inputs, evaluation_labels)) =
        get_data().unwrap();
    println!("{:?}", training_inputs);
}
