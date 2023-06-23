use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use memmap2::Mmap;
use pickle::DeOptions;
use serde_pickle as pickle;
use safetensors::{SafeTensors, Dtype};
use std::{
    fs::File,
    io::{Read, Seek},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// list the content
    List {
        /// The path to the tensor.
        tensor_file: PathBuf,
        /// if detailed listing is requested
        #[arg(short, long)]
        detailed: bool,
    },
    Convert {
        bin_file: PathBuf,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TorchTensorInfo {
    pub name: String,
    pub id: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl std::fmt::Display for TorchTensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TorchTensorInfo({},{},{:?},{})", self.name, self.id, self.dtype, display_shape(&self.shape))
    }
}

fn load(path: &Path) -> Result<Mmap> {
    let file = File::open(path).context(format!("Failed to read file: {}", path.display()))?;
    let buffer =
        unsafe { Mmap::map(&file).context(format!("Memory mapping file: {}", path.display()))? };
    Ok(buffer)
}

fn display_shape(shape: &[usize]) -> String {
    shape.iter().map(|&a| a.to_string()).collect::<Vec<_>>().join(" x ")
}

fn list(path: &Path, buffer: &[u8], detailed: bool) -> Result<()> {
    let tensors = SafeTensors::deserialize(buffer)
        .context(format!("Parsing {} as SafeTensor", path.display()))?;
    if detailed {
        tensors.tensors().iter().for_each(|(name, view)| {
            println!("{} - {:?} - {}", &name, &view.dtype(), display_shape(&view.shape()));
        });
    } else {
        tensors
            .names()
            .iter()
            .for_each(|name| println!("{}", &name));
    }
    Ok(())
}

fn get_pytorch_version<R>(bin_file: &Path, zip: &mut ZipArchive<R>) -> Result<(String, String)>
where
    R: Read + Seek,
{
    let data_pkl_file = zip
        .file_names()
        .find(|name| name.ends_with("/data.pkl"))
        .context(format!(
            "Missing (prefix)/data.pkl file from the Pickle file in {}!",
            bin_file.display()
        ))?;

    let version_file = data_pkl_file.replace("/data.pkl", "/version");
    let data_pkl_file = data_pkl_file.to_owned();
    println!("version file: {version_file} pickle: {data_pkl_file}");
    let mut version = zip
        .by_name(&version_file)
        .context(format!(
            "Missing {} - {} is not a pytorch file!",
            version_file,
            bin_file.display()
        ))?;
    let mut version_str = String::new();
    let readed: u64 = version.read_to_string(&mut version_str)?.try_into()?;
    assert_eq!(readed, version.size());
    Ok((version_str.replace("\n", ""), data_pkl_file))
}


fn analyze_pickle(pickle_obj: &pickle::Value) -> Result<String> {
    if let pickle::Value::Dict(dict) = pickle_obj {
        let values : Vec<_> = dict.into_iter().filter_map(|(key, value)| {
            if let pickle::HashableValue::String(key_string) = key {
                if let pickle::Value::Tuple(tuple) = value {
                    Some(TorchTensorInfo {
                        name: key_string.to_owned(),
                        id: "x".to_string(),
                        dtype: Dtype::F16,
                        shape: Vec::new(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }).collect();
        for tensor in values.into_iter() {
            println!("tensor key: {}", &tensor);
        }
        Ok("super".to_owned())
        // let key = pickle::HashableValue::String("_metadata".to_string());
        // if let Some(meta) = dict.get(&key) {
        //     if let pickle::Value::Dict(meta_dict) = meta {
        //         Ok("Super".to_owned())
        //     } else {
        //         Err(anyhow!("_metadata is not a dictionary!"))
        //     }
        // } else {
        //     Err(anyhow!("Missing _metadata key from the root!"))
        // }
    } else {
        Err(anyhow!("Root object is not a dictionary!"))
    }
}

fn get_pytorch_pickle<R>(bin_file: &Path, zip: &mut ZipArchive<R>, pickle_file_name: &str) -> Result<String>
where
    R: Read + Seek,
{
    let pickle_file = zip
        .by_name(pickle_file_name)
        .context(format!(
            "Missing {} - {} is not a pytorch file!",
            &pickle_file_name,
            bin_file.display()
        ))?;

    let decoded: pickle::Value = pickle::value_from_reader(pickle_file, DeOptions::new().replace_globals_to_tuples())
        .context(format!(
            "Unable to decode pickle file {} in {}!",
            &pickle_file_name,
            bin_file.display()
        ))?;
    let meta = analyze_pickle(&decoded)?;
    println!("analyze: '{}'", meta);
    let result = format!("{:#?}", decoded);
    println!("decoded: '{}'", result);
    Ok(result)
}

fn convert(bin_file: &Path) -> Result<()> {
    let file =
        File::open(bin_file).context(format!("Failed to read file: {}", bin_file.display()))?;
    let mut zip = zip::ZipArchive::new(file).context(format!(
        "Failed to open as a ZIP file: {}",
        bin_file.display()
    ))?;

    let (version, pickle_file) = get_pytorch_version(&bin_file, &mut zip)?;
    println!("Pytorch version: '{}' ({})", version, &pickle_file);
    zip.file_names().for_each(|name| println!("{}", &name));
    let pickled = get_pytorch_pickle(&bin_file, &mut zip, &pickle_file)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::List {
            detailed,
            tensor_file,
        }) => {
            let buffer = load(tensor_file)?;
            list(tensor_file, &buffer, *detailed)
        }
        Some(Commands::Convert { bin_file }) => convert(bin_file),
        None => Ok(()),
    }
}
