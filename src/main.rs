use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

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
}

fn load(path: &Path) -> Result<Mmap> {
    let file = File::open(path).context(format!("Failed to read file: {}", path.display()))?;
    let buffer =
        unsafe { Mmap::map(&file).context(format!("Memory mapping file: {}", path.display()))? };
    Ok(buffer)
}

fn list(path: &Path, buffer: &[u8], detailed: bool) -> Result<()> {
    let tensors = SafeTensors::deserialize(buffer)
        .context(format!("Parsing {} as SafeTensor", path.display()))?;
    if detailed {
        tensors.tensors().iter().for_each(|(name, view)| {
            let shapes: Vec<String> = view.shape().iter().map(|&a| a.to_string()).collect();
            println!("{} - {:?} - {}", &name, &view.dtype(), shapes.join(" x "));
        });
    } else {
        tensors
            .names()
            .iter()
            .for_each(|name| println!("{}", &name));
    }
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
        None => Ok(()),
    }
}
