use std::io::{Write, stdin, stdout};
mod embedder;

use embedder::EmbeddingTool;
use ollama_rs::{
    Ollama,
    coordinator::Coordinator,
    generation::{chat::ChatMessage, tools::implementations::*},
    models::ModelOptions,
};

//const MODEL: &str = "granite3.3";
//const MODEL: &str = "cogito:14b";
const MODEL: &str = "qwen3";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 2 || (args.get(1).is_some() && args[1] != "-d") {
        eprintln!("Usage: {} [-d] (to enable debugging)", args[0],);
        return Ok(());
    }

    let debug = args.get(1).is_some();
    let history = vec![];
    let ollama = Ollama::default();

    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history)
        .options(ModelOptions::default().num_ctx(16384))
        .add_tool(DDGSearcher::new())
        .add_tool(Scraper {})
        .add_tool(Calculator {})
        .add_tool(Scraper::new())
        .add_tool(DDGSearcher::new())
        .add_tool(EmbeddingTool {})
        .debug(debug);

    let stdin = stdin();
    let mut stdout = stdout();
    loop {
        stdout.write_all(b"\n> ")?;
        stdout.flush()?;

        let mut input = String::new();
        stdin.read_line(&mut input)?;

        let input = input.trim_end();
        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        let resp = coordinator
            .chat(vec![ChatMessage::user(input.to_string())])
            .await?;

        println!("{}", resp.message.content);
    }

    Ok(())
}
