use crate::MODEL;

use ollama_rs::Ollama;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::generation::tools::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{LazyLock, RwLock};

#[derive(Serialize, Deserialize)]
struct EmbeddingStore {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
}

static EMBEDDING_CACHE: LazyLock<RwLock<EmbeddingStore>> =
    LazyLock::new(|| RwLock::new(load_embeddings()));

#[derive(Deserialize, JsonSchema)]
pub struct Params {
    #[schemars(description = "The text to generate an embedding for.")]
    text: String,
}

pub struct EmbeddingTool {}

impl Tool for EmbeddingTool {
    type Params = Params;

    fn name() -> &'static str {
        "embedding"
    }

    fn description() -> &'static str {
        "Generates embeddings for input text."
    }

    async fn call(
        &mut self,
        parameters: Self::Params,
    ) -> Result<String, Box<dyn Error + Sync + Send>> {
        // Initialize Ollama client
        let client = Ollama::default();

        // Construct the embedding request
        let text = parameters.text.clone();
        let request = GenerateEmbeddingsRequest::new(MODEL.to_string(), parameters.text.into());

        // Send the request
        let response = client.generate_embeddings(request).await?;

        // Extract the embedding vector from the response
        let embedding_data = response.embeddings;

        EMBEDDING_CACHE
            .write()
            .unwrap()
            .embeddings
            .insert(text, embedding_data);

        Ok("Successfully generated embeddings".into())
    }
}

fn load_embeddings() -> EmbeddingStore {
    let store = EmbeddingStore {
        embeddings: HashMap::new(),
    };
    store
}
