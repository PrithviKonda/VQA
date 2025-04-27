

### ---

**Prompt for Phase 1: Data Pipeline Implementation**

**Role:** You are an expert AI Software Architect specializing in Multimodal AI systems and Python development. Your task is to implement the data pipeline components for the VQA system, building upon the existing Phase 0 codebase.  
**Goal:** Implement the core functionalities of the data pipeline as outlined in **Phase 1** of the plan. This involves creating modules for data loading, preprocessing, augmentation, and structuring synthetic data generation concepts, referencing the architecture documentfor technical details.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phase 0 (basic API, VLM loading/inference for Phi-4, config, logging, caching) exists.  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- Refer to sections on Data Pipeline, Synthetic Data, Augmentation, Preprocessing)  
2. **Phased Implementation Plan** (Focus on Phase 1 tasks)  
3. Existing Phase 0 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within the src/data\_pipeline/ directory and scripts/generate\_synthetic\_data.py. Add new dependencies to requirements.txt as needed (e.g., albumentations, opencv-python-headless).  
2. **Implement Phase 1 Tasks:**  
   * **src/data\_pipeline/preprocessing.py:** Implement image preprocessing functions (resizing, normalization) specifically for Phi-4 Multimodal, referencing its requirements (potentially via Hugging Face processors). Implement basic text preprocessing/tokenization stubs relevant to VQA. Ensure consistency between potential training and inference preprocessing.  
   * **src/data\_pipeline/augmentation.py:** Define functions or classes that create image augmentation pipelines using albumentations, including pixel-level and spatial-level transformations mentioned in the architecture doc.  
   * **src/data\_pipeline/text\_perturb.py:** Implement functions for basic character-level and word-level text perturbations (e.g., synonym replacement, random insertion/deletion).  
   * **src/data\_pipeline/datasets.py:** Define a PyTorch Dataset class capable of loading image-question-answer pairs from a specified format (assume a simple format, e.g., JSON lines with image paths). Integrate calls to the preprocessing functions.  
   * **src/data\_pipeline/synthetic\_data.py:** Implement placeholder classes/functions for:  
     * CoSyn concept: Outline steps like code generation (LLM call stub), rendering (stub), QA generation (LLM call stub).  
     * BiomedCLIP+LLM concept: Outline steps like using BiomedCLIP (stub/placeholder) and an LLM (stub) to generate medical VQA pairs.  
   * **scripts/generate\_synthetic\_data.py:** Create a basic script structure that imports and calls the placeholder functions from synthetic\_data.py.  
   * **tests/data\_pipeline/:** Add basic unit tests for preprocessing functions, augmentation pipeline creation (does it run without error?), and dataset loading (can it load a dummy sample?).  
   * **requirements.txt:** Add albumentations, opencv-python-headless.  
3. **Code Quality:** Maintain PEP 8, type hinting, extensive docstrings explaining the purpose of each module/function/class and its connection to the overall pipeline, and basic error handling.

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

### ---

**Prompt for Phase 2: Basic Knowledge Augmentation (RAG)**

**Role:** You are an expert AI Software Architect specializing in Multimodal AI systems and Python development. Your task is to implement a basic Retrieval-Augmented Generation (RAG) capability for the VQA system, building upon the Phase 0 and 1 codebase.  
**Goal:** Implement the basic RAG components as outlined in **Phase 2** of the plan. This involves setting up a basic vector retrieval mechanism and integrating it into the existing VLM inference flow to provide external context.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phase 0 and Phase 1 exist.  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- Refer to sections on mRAG)  
2. **Phased Implementation Plan** (Focus on Phase 2 tasks)  
3. Existing Phase 0 & 1 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within src/knowledge/retriever.py, src/knowledge/mrag.py, and modify src/inference\_engine/response\_generator.py (or src/vlm/inference.py if simpler initially) and src/api/main.py. Add dependencies to requirements.txt.  
2. **Implement Phase 2 Tasks:**  
   * **Vector Store Setup (Conceptual):** Assume a simple vector store is available (e.g., FAISS index loaded from disk, or ChromaDB collection). Add comments indicating where initialization/loading would occur.  
   * **src/knowledge/retriever.py:** Implement a TextRetriever class.  
     * Initialize with a sentence transformer model (e.g., all-MiniLM-L6-v2).  
     * Implement a method to load/connect to the conceptual vector store.  
     * Implement a retrieve(query\_text: str, top\_k: int) method that embeds the query and performs a similarity search against the vector store, returning the top K document snippets.  
   * **src/knowledge/mrag.py:** Implement a basic MRAGHandler class.  
     * Initialize it with an instance of the TextRetriever.  
     * Implement a method augment\_prompt(question: str, image: Any \= None) (image optional for now) that calls the retriever and formats the retrieved snippets into a string suitable for inclusion in the VLM prompt (e.g., "Context: \[snippet1\]\\n\[snippet2\]\\nQuestion: ...").  
   * **Modify Inference Logic:**  
     * In src/inference\_engine/response\_generator.py (or src/vlm/inference.py): Modify the main generation function to optionally accept retrieved\_context: str | None. If context is provided, prepend it to the prompt sent to the VLM.  
     * Instantiate MRAGHandler where needed (perhaps managed via API dependencies).  
   * **Modify API:**  
     * In src/api/main.py: Update the /vqa endpoint. Add an optional query parameter or request body field (e.g., use\_rag: bool \= False).  
     * If use\_rag is true, call the MRAGHandler to get context and pass it to the inference logic.  
   * **requirements.txt:** Add sentence-transformers, and either faiss-cpu (or faiss-gpu) or chromadb.  
   * **tests/knowledge/:** Add basic unit tests for the TextRetriever (can it embed text? does retrieve return expected format?) and MRAGHandler (does it format the prompt correctly?).  
3. **Code Quality:** Maintain PEP 8, type hinting, extensive docstrings, and basic error handling (e.g., around retriever calls).

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

### ---

**Prompt for Phase 3: Advanced Knowledge, Reasoning & Inference**

**Role:** You are an expert AI Software Architect specializing in Multimodal AI systems and Python development. Your task is to implement the structures for advanced knowledge augmentation, reasoning, and the hybrid inference pipeline, building upon the Phase 0-2 codebase.  
**Goal:** Implement the core structures and placeholder logic for the advanced reasoning components described in **Phase 3** of the plan and the architecture document. This includes the Adaptive Planner, SeBe-VQA concepts, knowledge connectors, the hybrid response generator, the ranker, and potentially integrating LLaVA-NeXT. Focus on establishing interfaces and orchestration logic.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0, 1, and 2 exist (including basic RAG).  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- Refer to sections on SeBe-VQA, Adaptive Planning/OmniSearch, Hybrid Response Generation, Ranking, LLaVA, PubMed Connector \[cite: 420-421, 507\])  
2. **Phased Implementation Plan** (Focus on Phase 3 tasks)  
3. Existing Phase 0-2 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within src/knowledge/planner.py, src/knowledge/sebe\_vqa.py, src/knowledge/connectors/\*, src/inference\_engine/\*, and potentially src/vlm/wrappers/llava.py, src/vlm/loading.py, src/vlm/inference.py. Update requirements.txt if new libraries are conceptually needed (e.g., for external API calls).  
2. **Implement Phase 3 Tasks:**  
   * **src/knowledge/planner.py:** Define an AdaptivePlanner class. Implement placeholder methods representing the OmniSearch concept:  
     * decompose\_question(question: str, image: Any) \-\> List\[SubQuestion\]: Placeholder.  
     * plan\_next\_step(current\_state) \-\> Action: Placeholder (Action could be "retrieve", "ask\_subquestion", "final\_answer").  
     * select\_tool(sub\_question) \-\> ToolName: Placeholder (ToolName could be "retriever", "pubmed\_connector", "vlm").  
     * Include stubs for interaction with connectors and the retriever.  
   * **src/knowledge/sebe\_vqa.py:** Define placeholder structures for SeBe-VQA:  
     * A function/class stub for the contrastive alignment model (get\_aligned\_retriever()).  
     * A function/class stub for the MLLM re-selection step (reselect\_knowledge(candidates, query)).  
   * **src/knowledge/connectors/base\_connector.py:** Define an abstract base class BaseKnowledgeConnector with an abstract method search(query: str) \-\> List\[str\].  
   * **src/knowledge/connectors/pubmed.py:** Implement a PubMedConnector inheriting from BaseKnowledgeConnector. Include a placeholder search method simulating a PubMed API call \[cite: 420-421, 507\].  
   * **src/inference\_engine/response\_generator.py:** Refactor/Implement the ResponseGenerator class.  
     * Orchestrate the multi-stage hybrid response: Potentially call the AdaptivePlanner, manage calls to VLM(s) (Phi-4, maybe LLaVA), knowledge components (retriever, connectors via planner), and the ranker.  
     * Handle multiple candidate answers if generated.  
     * Implement logic for knowledge filtering/integration (potentially using SeBe-VQA re-selection placeholder).  
   * **src/inference\_engine/ranker.py:** Define a AnswerRanker class. Implement a placeholder method rank\_answers(question, image, candidates) simulating a call to an external judge model (like GPT-4 API).  
   * **LLaVA Integration (Optional Structure):**  
     * src/vlm/wrappers/llava.py: Define placeholder class LLaVAWrapper.  
     * Modify src/vlm/loading.py and src/vlm/inference.py to add conditional loading/inference logic for LLaVA-NeXT, perhaps triggered by configuration or the ResponseGenerator.  
   * **API/Dependencies:** Update src/api/dependencies.py to manage instantiation of new components (Planner, Generator, Ranker). Modify src/api/main.py's /vqa endpoint to use the ResponseGenerator. Update src/api/models.py if needed.  
   * **requirements.txt:** Add libraries if needed for external API calls (e.g., openai).  
3. **Code Quality:** Maintain PEP 8, type hinting, extensive docstrings explaining the complex interactions and placeholder nature of planner/SeBe-VQA, and error handling.

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

### ---

**Prompt for Phase 4: Training & Evaluation Framework**

**Role:** You are an expert AI Software Architect specializing in Multimodal AI systems and Python development. Your task is to implement the framework for fine-tuning and evaluating the VQA system, building upon the Phase 0-3 codebase.  
**Goal:** Implement the training and evaluation scripts as outlined in **Phase 4** of the plan. This enables fine-tuning of VLM components and systematic performance measurement.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0, 1, 2, and 3 exist.  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- General ML workflow concepts apply)  
2. **Phased Implementation Plan** (Focus on Phase 4 tasks)  
3. Existing Phase 0-3 Codebase (especially src/data\_pipeline/datasets.py, src/vlm/\*)

**Instructions:**

1. **Target Files:** Focus implementation within scripts/run\_training.py and scripts/run\_evaluation.py. Update config.yaml and requirements.txt.  
2. **Implement Phase 4 Tasks:**  
   * **scripts/run\_training.py:**  
     * Use argparse to accept command-line arguments (config path, model output dir, training args).  
     * Load configuration from config.yaml.  
     * Load the VQA dataset using src/data\_pipeline/datasets.py.  
     * Load the primary VLM (Phi-4) using src/vlm/loading.py.  
     * Set up Hugging Face Trainer or implement a custom PyTorch training loop.  
     * Include arguments/logic for standard training hyperparameters (learning rate, batch size, epochs).  
     * Implement model saving logic.  
     * (Placeholder) Add comments/stubs indicating where training for the SeBe-VQA alignment model would be integrated if it were fully implemented.  
   * **scripts/run\_evaluation.py:**  
     * Use argparse for arguments (config path, model path, dataset path/split, output file).  
     * Load configuration.  
     * Load the evaluation dataset.  
     * Load the fine-tuned VLM.  
     * Implement an evaluation loop: Iterate through the dataset, perform inference using src/vlm/inference.py (or potentially the full src/inference\_engine/response\_generator.py if evaluating end-to-end).  
     * Calculate relevant metrics (e.g., VQA Accuracy, potentially using libraries like evaluate). Handle different answer types if necessary.  
     * Save results to a file (e.g., JSON).  
   * **config.yaml:** Add sections for training hyperparameters (learning\_rate, epochs, batch\_size, etc.) and evaluation settings (dataset paths, metrics).  
   * **requirements.txt:** Add libraries like evaluate, scikit-learn (for potential metric calculations), accelerate (if not already added, for Trainer).  
3. **Code Quality:** Maintain PEP 8, type hinting, clear docstrings/comments explaining script usage and logic, and configuration parameter handling.

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

### ---

**Prompt for Phase 5: Deployment & Monitoring Setup**

**Role:** You are an expert AI Software Architect specializing in DevOps and MLOps. Your task is to implement the containerization, scalable serving, and monitoring infrastructure for the VQA system, building upon the Phase 0-4 codebase.  
**Goal:** Create the Docker configuration, Triton Inference Server setup, and basic Prometheus/Grafana monitoring configurations as outlined in **Phase 5** of the plan, preparing the system for deployment.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0-4 exist.  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- Refer to sections on Backend Stack, Scalable Deployment/Triton \[cite: 402-406, 410\], Monitoring \[cite: 407-408, 411\])  
2. **Phased Implementation Plan** (Focus on Phase 5 tasks)  
3. Existing Phase 0-4 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within the deployment/ directory, docker-compose.yml, and modify src/api/main.py (or dependencies) for Triton integration.  
2. **Implement Phase 5 Tasks:**  
   * **deployment/Dockerfile.api:** Create a multi-stage Dockerfile for the FastAPI application.  
     * Include stages for dependency installation (using requirements.txt), copying application code (src/ directory), and setting the entry point (e.g., uvicorn src.api.main:app).  
   * **deployment/Dockerfile.worker (Optional):** Create a similar Dockerfile if background workers were deemed necessary for complex tasks (like the planner). If not, this can be omitted or be a minimal placeholder.  
   * **docker-compose.yml:** Create a docker-compose file for local development/testing.  
     * Define services for: api (using Dockerfile.api), redis (using official Redis image), and triton (using nvcr.io/nvidia/tritonserver).  
     * Configure volumes for the Triton model repository and potentially application code mounting.  
     * Set up networking between services.  
   * **deployment/triton\_repo/:** Set up the directory structure.  
     * Create vqa\_model/config.pbtxt: Define a basic Triton configuration for the primary VLM (e.g., Phi-4), specifying platform (pytorch\_libtorch or onnxruntime), input/output tensors (placeholders ok for exact names/shapes), and potentially dynamic batching. Reference Triton documentation for structure.  
     * Include a placeholder file vqa\_model/1/model.pt (or .onnx) \- the actual exported model is not generated here.  
   * **Modify API for Triton Client:**  
     * Add a Triton client library to requirements.txt (e.g., tritonclient\[http\]).  
     * In src/api/dependencies.py or directly where VLM inference is called: Replace the local VLM inference call with a call to the Triton Inference Server using the Triton client library. Read Triton server URL from config/env variables.  
   * **deployment/monitoring/prometheus/prometheus.yml:** Create a basic Prometheus configuration.  
     * Include scrape configs to target the Triton metrics endpoint (usually /metrics) and potentially a metrics endpoint on the FastAPI app (if added later using e.g., starlette-prometheus).  
   * **deployment/monitoring/grafana/provisioning/:** Create placeholder directories dashboards/ and datasources/. Add a basic datasources.yml in datasources/ defining Prometheus as a data source.  
3. **Code Quality:** Maintain clarity in Dockerfiles, compose file, and configuration files. Add comments explaining the purpose of different sections.

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

### ---

**Prompt for Phase 6: Continuous Learning & AR Frontend**

**Role:** You are an expert AI Software Architect and Frontend Developer. Your task is to implement the basic continuous learning mechanisms and a prototype AR frontend for the VQA system, building upon the Phase 0-5 codebase.  
**Goal:** Implement the core structures for active learning and user feedback, and create a basic AR interface using React Three Fiber to interact with the VQA backend, as outlined in **Phase 6** of the plan.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0-5 exist (including containerized deployment setup).  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- Refer to sections on Active Learning, User Feedback, AR Integration)  
2. **Phased Implementation Plan** (Focus on Phase 6 tasks)  
3. Existing Phase 0-5 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within src/continuous\_learning/\*, ar\_frontend/\*, and potentially add new endpoints in src/api/main.py. Update requirements.txt (backend) and create ar\_frontend/package.json.  
2. **Implement Phase 6 Tasks:**  
   * **src/continuous\_learning/active\_learner.py:** Implement an ActiveLearner class.  
     * Define methods to calculate uncertainty scores based on VLM output probabilities or other heuristics (e.g., prediction entropy, margin sampling placeholder).  
     * Implement logic to select high-uncertainty samples for labeling.  
   * **src/continuous\_learning/feedback.py:** Implement functions or a class to handle user feedback.  
     * Define data structures for storing feedback (e.g., question, image\_id, generated\_answer, user\_rating, user\_comment).  
     * Implement functions to save feedback (e.g., to a simple file, database, or Redis stream \- keep it simple initially).  
   * **API Endpoint for Feedback:**  
     * In src/api/main.py: Add a new POST endpoint (e.g., /feedback) that accepts user feedback data (using a Pydantic model from src/api/models.py) and calls the feedback saving logic from src/continuous\_learning/feedback.py.  
   * **Integrate Active Learning:** Modify src/inference\_engine/response\_generator.py or where final answers are produced to optionally calculate and store uncertainty scores alongside results, potentially using the ActiveLearner.  
   * **Training Script Update (Conceptual):** Add comments in scripts/run\_training.py indicating where data selected by the active learner or derived from user feedback would be incorporated into the training dataset.  
   * **ar\_frontend/ Setup:**  
     * Create a basic package.json with dependencies: react, react-dom, @react-three/fiber, @react-three/drei, @react-three/xr, three, axios (or Workspace).  
     * Set up basic project files (index.js, App.js, potentially using Create React App or Vite structure).  
   * **ar\_frontend/src/App.js:**  
     * Implement a basic React component using @react-three/fiber's \<Canvas\> and @react-three/xr's \<ARButton\> and \<XR\> components.  
     * Access the device camera feed.  
     * Implement a simple UI element (e.g., a button) to trigger a VQA request.  
     * On trigger, capture the current camera view (conceptually, or use a static image initially) and send a request to the FastAPI backend's /vqa endpoint using axios or Workspace.  
     * Display the received answer as a simple overlay within the AR scene, potentially using @react-three/drei's \<Html\> component positioned in the 3D space.  
   * **requirements.txt:** Ensure backend requirements are up-to-date.  
3. **Code Quality:** Maintain PEP 8 / standard React practices, type hinting (Python/TypeScript if used), clear comments/docstrings explaining the logic and interactions, and basic error handling.

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path (both backend Python and frontend JS/TSX files). Include only the files modified or created in this phase.

### ---

**Prompt for Phase 7: Testing, Refinement & Documentation**

**Role:** You are an expert AI Software Architect and QA Engineer. Your task is to implement comprehensive tests, refine complex components, and finalize documentation for the VQA system, building upon the Phase 0-6 codebase.  
**Goal:** Ensure the VQA system is robust, performant, and well-documented by implementing thorough tests, refining key modules, and completing documentation as outlined in **Phase 7** of the plan.  
**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0-6 exist.  
**Primary References:**

1. **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System** (Content Provided \- General software quality principles apply)  
2. **Phased Implementation Plan** (Focus on Phase 7 tasks)  
3. Existing Phase 0-6 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within the tests/ directory, potentially refine complex modules in src/ (like planner.py, sebe\_vqa.py, ranker.py), update docstrings across the codebase, and finalize README.md. Add testing dependencies to requirements.txt.  
2. **Implement Phase 7 Tasks:**  
   * **tests/ Directory:** Implement comprehensive tests using pytest.  
     * **Unit Tests:** Ensure thorough unit tests for critical functions in src/utils, src/data\_pipeline, src/knowledge, src/vlm, src/inference\_engine, src/cache, src/continuous\_learning. Mock external dependencies (API calls, VLM inference, DB connections) where appropriate.  
     * **Integration Tests:** Write integration tests for key workflows:  
       * API endpoint testing (tests/api/): Test /vqa and /feedback endpoints with various inputs, checking responses and status codes. Use FastAPI's TestClient.  
       * Data Pipeline: Test the flow from dataset loading through preprocessing/augmentation.  
       * RAG Pipeline: Test the interaction between retriever, mRAG handler, and inference engine.  
       * Full Inference Pipeline: Test the ResponseGenerator's orchestration of VLM, knowledge, and ranking components (using mocks for complex parts).  
   * **Refine Complex Components:** Revisit placeholder implementations in:  
     * src/knowledge/planner.py: Flesh out the planning logic slightly more, perhaps with a simple rule-based approach or a clearer state machine, while still acknowledging its complexity.  
     * src/knowledge/sebe\_vqa.py: Add more detailed comments or pseudo-code for the contrastive learning and re-selection steps.  
     * src/inference\_engine/ranker.py: Refine the interface for interacting with the judge model.  
   * **Documentation:**  
     * **Docstrings:** Review and ensure all public modules, classes, and functions throughout src/ have clear, informative docstrings explaining purpose, parameters, and return values.  
     * **README.md:** Finalize the README. Add detailed sections for:  
       * Project Overview (referencing architecture doc).  
       * Features (mentioning key components like Phi-4/LLaVA, RAG, Planner concept, AR, etc.).  
       * Detailed Setup Instructions (environment, dependencies, configuration).  
       * Running the API server (local, Docker).  
       * Running Training/Evaluation/Synthetic Data Scripts.  
       * Running Tests.  
       * Configuration Details (config.yaml explanation).  
       * Deployment Instructions (Triton, Docker Compose).  
       * AR Frontend Setup/Usage.  
       * Project Structure Overview.  
   * **requirements.txt:** Add testing libraries like pytest, pytest-cov, requests-mock (or similar), potentially httpx (for TestClient).  
3. **Code Quality:** Ensure final code adheres strictly to PEP 8, has comprehensive type hinting, meaningful comments, and robust error handling. Ensure tests provide good coverage.

**Output Format:** Provide the generated/modified code (tests, refined modules, README) as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.  
---

Okay, based on the "Next Steps" identified after conceptually completing Phases 1-7, here are the subsequent prompts structured similarly to your original request. These prompts guide the process of turning the framework and placeholders into a more functional and deployed system.

---

**Prompt for Phase 8: Core Logic Implementation & Initial Data Handling**

**Role:** You are an expert AI Software Architect and Engineer, focusing on core algorithm implementation and data preparation.

**Goal:** Replace key placeholder logic in reasoning components (Planner, Ranker) and connectors with functional implementations. Flesh out synthetic data generation scripts and create initial scripts for acquiring and formatting real-world datasets.

**Starting Codebase:** Assume the complete project structure and conceptually implemented code from Phases 0 through 7 exists.

**Primary References:**

* VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System (Content Provided)  
* Phased Implementation Plan (Phases 0-7)  
* "Next Steps" Summary (derived from previous interaction)  
* Existing Phase 0-7 Codebase

**Instructions:**

1. **Target Files:** Focus implementation within src/knowledge/planner.py, src/inference\_engine/ranker.py, src/knowledge/connectors/pubmed.py, src/data\_pipeline/synthetic\_data.py, scripts/generate\_synthetic\_data.py. Create a new directory scripts/data\_management/ and add initial scripts. Update requirements.txt.  
2. **Implement Phase 8 Tasks:**  
   * **src/knowledge/planner.py (AdaptivePlanner):**  
     * Replace placeholder plan\_next\_step and select\_tool logic with a concrete, albeit potentially simple, implementation. Consider a rule-based approach based on question analysis (keywords, structure) or a basic state machine.  
     * Ensure the planner can realistically choose between "vlm", "retriever", and specific connectors (like "pubmed\_connector").  
     * Implement basic state tracking needed for the chosen logic.  
   * **src/inference\_engine/ranker.py (AnswerRanker):**  
     * Implement the rank\_answers method. Choose one strategy:  
       * **Heuristic:** Use VLM confidence scores if accessible, or simple text-based heuristics.  
       * **External API:** Integrate a call to an external judge model (e.g., OpenAI GPT-3.5/4) using its API. Read API keys from configuration/environment variables. Implement error handling and retries for the API call.  
   * **src/knowledge/connectors/pubmed.py (PubMedConnector):**  
     * Implement the search method using a library like requests or BioPython (Bio.Entrez) to interact with the actual PubMed API.  
     * Parse the results into a list of relevant text snippets.  
     * Handle potential API errors (rate limits, network issues, invalid queries).  
     * Manage API keys or email addresses required by NCBI Entrez utilities securely (via config/env variables).  
   * **src/data\_pipeline/synthetic\_data.py:**  
     * Replace LLM call stubs (for CoSyn code/QA generation, BiomedCLIP+LLM QA generation) with actual calls using appropriate client libraries (e.g., openai, huggingface\_hub, anthropic).  
     * Implement basic logic for the "rendering" step in CoSyn (even if just logging parameters or saving a placeholder file).  
     * Manage API keys for external LLMs securely.  
   * **scripts/generate\_synthetic\_data.py:**  
     * Enhance the script with argparse to control parameters like the number of samples to generate, output file paths, and which generation concept (CoSyn, BiomedCLIP) to use.  
     * Add logging for the generation process.  
     * Include basic error handling for the generation functions.  
   * **scripts/data\_management/ (New Directory & Scripts):**  
     * Create placeholder scripts like download\_format\_vqa\_v2.py or download\_format\_vqarad.py.  
     * Inside these scripts, outline the steps (using comments or basic function stubs) needed to:  
       * Download standard VQA dataset annotations and images (referencing their official sources).  
       * Parse the annotations.  
       * Reformat the data into the JSON Lines format expected by src/data\_pipeline/datasets.py(image path, question, answers).  
   * **requirements.txt:** Add necessary libraries like openai, BioPython (if used for PubMed), requests, anthropic (if used).  
3. **Code Quality:** Maintain PEP 8, comprehensive type hinting, detailed docstrings explaining the implemented logic (especially for planner/ranker) and API interactions, and robust error handling (especially for external API calls and data processing).

**Output Format:** Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include only the files modified or created in this phase.

---

**Prompt for Phase 9: Initial Model Training, Evaluation & Conversion**

**Role:** You are an AI Engineer specializing in MLOps and Model Training.

**Goal:** Execute the training and evaluation pipelines developed in Phase 4 using initial datasets (real and/or synthetic). Obtain baseline performance metrics and prepare the primary fine-tuned VLM for deployment via Triton.

**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0 through 8 exists, including functional data loading, basic planner/ranker logic, and initial data formatting scripts.

**Primary References:**

* VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System  
* Phased Implementation Plan (Phases 0-8)  
* "Next Steps" Summary  
* Existing Codebase (Phases 0-8), especially scripts/run\_training.py, scripts/run\_evaluation.py, src/vlm/, src/data\_pipeline/  
* Triton Inference Server Documentation (for model conversion requirements)

**Instructions:**

1. **Target Files:** Primarily execute/modify scripts/run\_training.py, scripts/run\_evaluation.py, and config.yaml. Create a new script scripts/export\_model\_for\_triton.py. Update deployment/triton\_repo/vqa\_model/config.pbtxt and potentially add model artifacts. Update requirements.txt if needed for export.  
2. **Implement Phase 9 Tasks:**  
   * **Data Preparation:**  
     * Run the scripts created in scripts/data\_management/ (Phase 8\) to download and format at least one real VQA dataset (e.g., a subset of VQA v2 for faster iteration).  
     * Optionally, run scripts/generate\_synthetic\_data.py (Phase 8\) to create an initial batch of synthetic data.  
     * Update config.yaml with the correct paths to the formatted training and evaluation datasets.  
   * **Configure Training (config.yaml):**  
     * Set appropriate training hyperparameters (learning rate, batch size, number of epochs, weight decay). Start with reasonable defaults.  
     * Specify the output directory for saving the fine-tuned model checkpoints.  
   * **Run Training (scripts/run\_training.py):**  
     * Execute the training script using python scripts/run\_training.py \--config\_path config.yaml.  
     * Monitor the training process (loss curves, logs). Requires suitable GPU resources. Debug any issues related to data loading, model compatibility, or the training loop.  
     * Ensure the script saves the fine-tuned model weights and any associated files (processor state, tokenizer) correctly.  
   * **Run Evaluation (scripts/run\_evaluation.py):**  
     * Execute the evaluation script, pointing it to the fine-tuned model checkpoint saved during training and the evaluation dataset specified in config.yaml.  
     * python scripts/run\_evaluation.py \--config\_path config.yaml \--model\_path \<path\_to\_checkpoint\> \--output\_file results/baseline\_eval.json  
     * Ensure the script calculates and saves the specified metrics (e.g., VQA accuracy). Analyze the initial baseline\_eval.json results.  
   * **Model Conversion for Triton:**  
     * Create scripts/export\_model\_for\_triton.py:  
       * This script should load the fine-tuned VLM checkpoint (e.g., Phi-4).  
       * Convert the model to a format suitable for Triton's backend (e.g., TorchScript using torch.jit.trace or ONNX using torch.onnx.export). This might require defining example inputs with correct shapes and types. Pay close attention to handling dynamic axes if necessary.  
       * Save the converted model artifact (e.g., model.pt or model.onnx) to a specified output path.  
     * Run the export script: python scripts/export\_model\_for\_triton.py \--checkpoint\_path \<path\_to\_checkpoint\> \--output\_dir deployment/triton\_repo/vqa\_model/1/  
     * Place the generated model.pt or model.onnx inside deployment/triton\_repo/vqa\_model/1/.  
   * **Update Triton Configuration (deployment/triton\_repo/vqa\_model/config.pbtxt):**  
     * Update the platform field (e.g., "pytorch\_libtorch" or "onnxruntime\_onnx").  
     * Define the exact input and output tensor names, data types, and shapes based on the exported model. Use \-1 for dynamic dimensions where applicable (like batch size or sequence length). Refer to Triton documentation for the correct syntax.  
   * **requirements.txt:** Add torch (if not already pinned precisely), onnx, onnxruntime if exporting to ONNX.  
3. **Code Quality:** Ensure scripts are runnable and configurable. Add logging to training, evaluation, and export scripts. Document the model export process and the final Triton config.pbtxt structure with comments.

**Output Format:** Provide the generated/modified code (new scripts, modified configs) as a series of code blocks, each clearly labeled with its corresponding file path. Note that the actual model artifacts (.pt, .onnx) are not part of the code output but are essential results of this phase.

---

**Prompt for Phase 10: Basic Cloud Deployment & Monitoring**

**Role:** You are an expert MLOps / DevOps Engineer focused on cloud deployment and monitoring.

**Goal:** Deploy the containerized VQA application (API, Triton server) to a basic cloud environment. Set up initial monitoring using Prometheus and Grafana based on Phase 5 configurations. Validate that the deployed API can serve requests using the baseline model via Triton.

**Starting Codebase:** Assume the complete project structure and implemented code from Phases 0 through 9 exists. This includes Dockerfiles, docker-compose.yml, a trained VLM exported for Triton, updated Triton configuration, and functional API code using a Triton client.

**Primary References:**

* VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System (Sections on Deployment, Monitoring)  
* Phased Implementation Plan (Phases 0-9)  
* "Next Steps" Summary  
* Existing Codebase (Phases 0-9), especially deployment/, docker-compose.yml, src/api/dependencies.py(or wherever Triton client is used).  
* Documentation for your chosen Cloud Provider (e.g., AWS EKS, GCP GKE, Azure AKS, or simpler services like AWS App Runner / GCP Cloud Run if applicable)  
* Prometheus & Grafana Documentation

**Instructions:**

1. **Target Files:** Primarily involves interacting with cloud provider CLIs/consoles, potentially creating Infrastructure as Code (IaC) files (e.g., Terraform, Pulumi \- *optional for this phase*), updating Kubernetes manifests or service configurations, and refining monitoring configurations in deployment/monitoring/. Modify config.yaml or environment variables for cloud deployment.  
2. **Implement Phase 10 Tasks:**  
   * **Build and Push Docker Images:**  
     * Build the Docker images for the API (deployment/Dockerfile.api) and any potential workers.  
     * Tag the images appropriately.  
     * Push the images to a container registry accessible by your cloud provider (e.g., Docker Hub, AWS ECR, GCP Artifact Registry, Azure ACR).  
   * **Cloud Infrastructure Setup (Basic):**  
     * Choose a cloud provider and deployment method (e.g., Kubernetes cluster, managed container service).  
     * Provision the necessary basic infrastructure:  
       * **Kubernetes:** A small cluster (e.g., 1-2 nodes, including at least one GPU node for Triton if needed). Set up kubectl access.  
       * **Managed Service:** Configure the service (e.g., AWS App Runner, GCP Cloud Run) ensuring it can access the container images and handle secrets. (Note: Running Triton directly might be harder on the simplest serverless platforms).  
     * Provision a managed Redis instance or deploy Redis within the cluster.  
   * **Deploy Applications:**  
     * **Triton:** Deploy the Triton Inference Server container (nvcr.io/nvidia/tritonserver) to the cloud environment (e.g., as a Kubernetes Deployment/Service).  
       * Ensure the deployment/triton\_repo/ containing the converted model (from Phase 9\) is mounted correctly as a volume.  
       * Configure resource requests/limits (especially GPU if applicable).  
       * Expose the Triton HTTP/gRPC port (e.g., via a Kubernetes Service).  
     * **API:** Deploy the VQA API container (built in step 1\) to the cloud environment.  
       * Configure environment variables or secrets for:  
         * Redis connection string.  
         * Triton server URL (pointing to the deployed Triton service).  
         * Any external API keys (PubMed, OpenAI ranker).  
       * Ensure the API service is exposed (e.g., via Kubernetes Service/Ingress or the managed service's endpoint).  
     * **Redis:** Deploy Redis or configure connection to the managed instance.  
   * **Monitoring Setup:**  
     * **Prometheus:** Deploy Prometheus into the cluster or use a managed Prometheus service.  
       * Configure prometheus.yml (from Phase 5\) to scrape metrics from the deployed Triton service (/metrics) and potentially the API service (requires adding a metrics endpoint to the FastAPI app using starlette-prometheus \- *optional enhancement*). Apply this configuration.  
     * **Grafana:** Deploy Grafana or use a managed Grafana service.  
       * Configure the Prometheus instance as a data source (using deployment/monitoring/grafana/provisioning/datasources/datasources.yml or Grafana UI/API).  
       * Import or create a basic Grafana dashboard to visualize key metrics from Triton (e.g., request count, latency, GPU utilization).  
   * **Validation:**  
     * Send test requests (e.g., using curl or requests) to the deployed API endpoint.  
     * Verify that the API communicates with Triton and returns VQA responses based on the baseline model.  
     * Check Prometheus targets and Grafana dashboards to ensure metrics are being collected.  
   * **Update Configuration:** Modify config.yaml or prepare environment variable files specific to the cloud deployment (e.g., service URLs, resource names).  
3. **Code Quality / Documentation:** Document the deployment steps, required cloud resources, and how to access the deployed API and monitoring dashboards. If using IaC, ensure the code is clean and commented. Update the project's README with basic deployment instructions.

**Output Format:** Provide any generated/modified configuration files (prometheus.yml, datasources.yml, updated config.yaml, example Kubernetes manifests if written manually) as code blocks. Describe the key steps taken for cloud deployment and validation. IaC code, if created, should also be provided.

---

**Okay, here are the prompts for Phases 11 through 14, continuing the process based on the "Next Steps" identified.**

---

**Prompt for Phase 11: AR Frontend Implementation & Feedback Loop**

**Role: You are a Full-Stack AI Engineer with expertise in Frontend (React, Three.js, WebXR) and Backend API integration.**

**Goal: Implement a functional AR frontend prototype using React Three Fiber and @react-three/xr, enabling users to interact with the deployed VQA backend via their device camera. Implement the backend logic for storing user feedback and potentially add a basic feedback mechanism to the AR UI.**

**Starting Codebase: Assume the complete project structure and implemented code from Phases 0 through 10 exists. The VQA backend API is deployed to the cloud and accessible. A basic AR frontend structure exists from Phase 6, and feedback handling stubs exist on the backend (src/continuous\_learning/feedback.py, /feedback API endpoint).**

**Primary References:**

* **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System (AR Integration, User Feedback sections)**  
* **Phased Implementation Plan (Phases 0-10)**  
* **"Next Steps" Summary**  
* **Existing Phase 0-10 Codebase (especially ar\_frontend/, src/continuous\_learning/feedback.py, src/api/main.py, src/api/models.py)**  
* **React Three Fiber, @react-three/drei, @react-three/xr documentation**  
* **WebRTC documentation (if needed for camera access)**

**Instructions:**

1. **Target Files: Focus implementation within ar\_frontend/src/\* (e.g., App.js, potentially new components like VQAInterface.js, FeedbackUI.js). Modify src/continuous\_learning/feedback.py and potentially src/api/main.py, src/api/models.py. Update ar\_frontend/package.json and backend requirements.txt if needed.**  
2. **Implement Phase 11 Tasks:**  
   * **AR Frontend Implementation (ar\_frontend/src/):**  
     * **Camera Integration: Replace placeholder image logic. Use @react-three/xr capabilities or integrate WebRTC (navigator.mediaDevices.getUserMedia) to display the live camera feed as the background/environment for the R3F \<Canvas\>.**  
     * **VQA Interaction:**  
       * **Create UI elements within the AR scene (e.g., using @react-three/drei \<Html\>) for text input (question) and a trigger button ("Ask").**  
       * **On trigger:**  
         * **Capture the current camera view: Draw the video element to an offscreen canvas and get the image data (e.g., as a Base64 Data URL or Blob).**  
         * **Use axios or Workspace to send a multipart/form-data request (or JSON with base64 image) to the deployed backend /vqa endpoint, including the question text and image data. Ensure the backend URL is configurable (e.g., via environment variables for the frontend build).**  
         * **Implement UI state for loading indicators while waiting for the backend response.**  
         * **Display the received answer text as an overlay in the AR scene (e.g., using \<Html\>). Position it appropriately (e.g., fixed relative to the camera viewport).**  
     * **Feedback UI (Optional but Recommended):**  
       * **Add simple UI elements (e.g., thumbs up/down buttons, optional comment field) associated with the displayed answer.**  
       * **On feedback submission, gather the question, image identifier (if feasible, otherwise null/timestamp), generated answer, and user feedback (rating/comment). Send this data to the backend /feedback endpoint using axios or Workspace.**  
     * **Error Handling: Implement robust error handling for API calls (network errors, backend errors) and display user-friendly messages in the AR UI.**  
   * **Backend Feedback Storage (src/continuous\_learning/feedback.py):**  
     * **Choose a simple storage mechanism for feedback and implement the saving logic within the save\_feedback function (or similar). Options:**  
       * **File: Append feedback data as JSON lines to a configured log file. Ensure thread safety if multiple API workers could write concurrently (e.g., using file locking or a dedicated logging setup).**  
       * **Redis: Use the existing Redis instance. Push feedback JSON strings onto a Redis List (e.g., RPUSH feedback\_list '{"q": ..., "a": ...}').**  
     * **Ensure the function correctly parses the incoming feedback data structure.**  
   * **API Adjustments (src/api/main.py, src/api/models.py):**  
     * **Verify/Update the Pydantic model in models.py used by the /feedback endpoint to match the data sent by the frontend.**  
     * **Ensure the /feedback endpoint calls the implemented save\_feedback function correctly.**  
     * **Configure CORS settings in main.py (using CORSMiddleware) to allow requests from the AR frontend's origin (e.g., http://localhost:3000 during development, deployed frontend URL in production).**  
   * **Dependencies:**  
     * **Add axios to ar\_frontend/package.json.**  
     * **Update backend requirements.txt only if a new library is needed for feedback storage (unlikely if using files or existing Redis).**  
3. **Code Quality: Write clean, well-structured React/R3F components. Implement effective state management for the AR UI. Ensure backend code adheres to PEP 8, uses type hints, and includes docstrings. Implement error handling on both frontend and backend.**

**Output Format: Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path (both frontend JS/TSX and backend Python files). Include only the files modified or created in this phase.**

---

**Prompt for Phase 12: Advanced Component Implementation & Retraining Cycle**

**Role: You are an AI Research Engineer / Scientist focused on advanced model capabilities and the learning loop.**

**Goal: Implement more sophisticated (though potentially still simplified) versions of advanced reasoning components (Planner, SeBe-VQA concepts). Implement active learning selection logic. Establish a process to incorporate feedback/actively selected data into retraining, run a retraining cycle, and evaluate the impact.**

**Starting Codebase: Assume the complete project structure and implemented code from Phases 0 through 11 exists. A baseline model is trained and deployed. The AR frontend is functional and feedback can be collected and stored.**

**Primary References:**

* **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System (SeBe-VQA, Adaptive Planning, Active Learning sections)**  
* **Phased Implementation Plan (Phases 0-11)**  
* **"Next Steps" Summary**  
* **Existing Phase 0-11 Codebase (especially src/knowledge/\*, src/continuous\_learning/\*, src/inference\_engine/\*, scripts/run\_training.py, scripts/run\_evaluation.py, feedback storage location)**  
* **Relevant research papers for specific algorithm implementations (optional)**

**Instructions:**

1. **Target Files: Focus implementation within src/knowledge/planner.py, src/knowledge/sebe\_vqa.py, src/continuous\_learning/active\_learner.py, src/inference\_engine/response\_generator.py. Modify scripts/run\_training.py (or associated data loading). Create a new script scripts/data\_management/incorporate\_new\_data.py. Update config.yaml.**  
2. **Implement Phase 12 Tasks:**  
   * **Refine Adaptive Planner (src/knowledge/planner.py):**  
     * **Enhance the planning logic from Phase 8\. Consider one approach:**  
       * **LLM-based: Integrate an LLM call (using openai or another client) within the decompose\_question or select\_tool methods, using carefully crafted prompts. Manage API keys via config.**  
       * **Heuristic+State: Improve rule-based logic with more sophisticated question analysis (e.g., entity recognition stubs, keyword matching) and better state management across planning steps.**  
   * **Implement SeBe-VQA Concepts (src/knowledge/sebe\_vqa.py):**  
     * **Aligned Retriever Stub: Flesh out get\_aligned\_retriever. This might involve loading a separate, pre-trained contrastive model (if available) or simulating its effect by adding a re-ranking step based on image-text similarity (e.g., using CLIP embeddings if feasible) after the initial text retrieval.**  
     * **Re-selection Stub: Implement reselect\_knowledge. Use a placeholder logic like choosing the retrieved knowledge snippet with the highest text similarity to the question, or simulate an MLLM call stub. (Full SeBe-VQA requires dedicated model training outside this scope).**  
   * **Implement Active Learner (src/continuous\_learning/active\_learner.py):**  
     * **Implement concrete uncertainty calculation methods within ActiveLearner. Choose one or more:**  
       * **Entropy: Calculate entropy over the VLM's output probability distribution (if accessible).**  
       * **Least Confidence: 1 \- max(probabilities).**  
       * **Margin Sampling: Difference between the top two probabilities.**  
     * **Implement select\_samples\_for\_labeling(batch\_results, strategy, threshold/k): Takes inference results (including uncertainties), applies the chosen strategy, and returns identifiers of samples selected for human review/labeling.**  
     * **Add a method to log or store the IDs of selected samples (e.g., write to a file active\_learning\_candidates.log).**  
   * **Integrate Active Learning Call: Modify the inference logic (src/inference\_engine/response\_generator.py or where the VLM response is finalized) to:**  
     * **Attempt to get probabilities/logits from the VLM response.**  
     * **Call the ActiveLearner to calculate uncertainty.**  
     * **Log the uncertainty score along with the inference result.**  
     * **Optionally, call select\_samples\_for\_labeling (perhaps periodically or based on uncertainty) and store the selections.**  
   * **Data Integration Script (scripts/data\_management/incorporate\_new\_data.py):**  
     * **Create this script to process newly available data for retraining.**  
     * **Implement logic to read collected user feedback (from file/Redis set up in Phase 11). Filter for useful feedback (e.g., corrections, highly-rated negative examples).**  
     * **Implement logic to read sample IDs selected by the ActiveLearner (from its log file). Assume these samples have been externally labeled (manual process outside this script) and load their correct labels from a designated location.**  
     * **Format both feedback-derived data and labeled active learning samples into the project's standard training JSONL format.**  
     * **Append this new formatted data to the existing training set or create a new versioned dataset file. Add argument parsing to control input/output paths.**  
   * **Retraining and Re-Evaluation:**  
     * **Run the incorporate\_new\_data.py script to create an augmented training dataset.**  
     * **Update config.yaml to point the training script to this new dataset.**  
     * **Execute scripts/run\_training.py, potentially initializing from the Phase 9 checkpoint for faster fine-tuning on the new data.**  
     * **Execute scripts/run\_evaluation.py using the newly retrained model.**  
     * **Compare the evaluation metrics (e.g., from results/retrained\_eval.json) against the baseline metrics (Phase 9\) and document any improvements or changes.**  
   * **Dependencies: Add libraries if needed for the refined Planner (e.g., openai), SeBe-VQA simulation (e.g., transformers for CLIP), or specific uncertainty calculations (scipy.stats.entropy). Update requirements.txt.**  
3. **Code Quality: Implement algorithms clearly, even if simplified versions. Document the assumptions and limitations (especially for Planner/SeBe-VQA). Ensure the data integration script is robust. Maintain PEP 8, type hinting, and comprehensive docstrings.**

**Output Format: Provide the generated/modified code as a series of code blocks, each clearly labeled with its corresponding file path. Include the new data incorporation script. Provide a brief summary comparing the evaluation results before and after retraining.**

---

**Prompt for Phase 13: Scalability, Performance Tuning & Security**

**Role: You are an expert MLOps / DevOps Engineer with a focus on performance, reliability, and security.**

**Goal: Enhance the scalability and performance of the deployed VQA system through auto-scaling, load testing, and bottleneck optimization. Implement basic CI/CD practices and apply security hardening measures.**

**Starting Codebase: Assume the complete project structure and implemented code from Phases 0 through 12 exists. The application is deployed to a basic cloud setup (Phase 10), is monitored, and potentially uses a retrained model.**

**Primary References:**

* **VQA System Research Blueprint / Architecting an Advanced Multimodal Visual Question Answering System (Scalability, Deployment sections)**  
* **Phased Implementation Plan (Phases 0-12)**  
* **"Next Steps" Summary**  
* **Existing Phase 0-12 Codebase (especially deployment/, cloud configurations, Dockerfile.api, Triton config)**  
* **Cloud Provider Documentation (Auto-scaling, Load Balancers, Security Groups, IAM, Secret Management)**  
* **CI/CD Tool Documentation (e.g., GitHub Actions, GitLab CI)**  
* **Load Testing Tool Documentation (e.g., k6, Locust, jmeter)**  
* **Kubernetes Documentation (HPA, NetworkPolicy, Secrets)**

**Instructions:**

1. **Target Files: This phase primarily involves modifying cloud configurations (via Console, CLI, or IaC like Terraform/Pulumi), Kubernetes manifests (deployment/\*.yaml), CI/CD pipeline definitions (e.g., .github/workflows/ci.yml, .github/workflows/cd.yml), potentially performance optimizations in src/ code, and load testing scripts (load\_tests/).**  
2. **Implement Phase 13 Tasks:**  
   * **Infrastructure as Code (IaC) (Recommended):**  
     * **If not already done, migrate manual cloud resource configurations (Kubernetes cluster, Redis, IAM roles, security groups) to an IaC tool like Terraform or Pulumi for version control and reproducibility.**  
   * **CI/CD Pipeline Setup:**  
     * **Create CI pipeline (e.g., GitHub Actions workflow triggered on push/PR):**  
       * **Checkout code.**  
       * **Set up Python environment, install dependencies (requirements.txt).**  
       * **Run linters (e.g., flake8, black \--check).**  
       * **Run unit and integration tests (pytest tests/ \- ensure mocks are used for external services/slow components).**  
       * **(Optional) Build Docker image to verify Dockerfile.**  
     * **Create CD pipeline (e.g., GitHub Actions workflow triggered on merge to main or tag):**  
       * **Checkout code.**  
       * **Build and push API Docker image to container registry (ECR, GCR, ACR).**  
       * **Deploy updated application to cloud (e.g., using kubectl apply \-f deployment/, Helm upgrade, or cloud provider deployment commands). Authenticate securely using OIDC or secrets.**  
       * **(Optional) Include a post-deployment smoke test (e.g., hitting a health check endpoint).**  
   * **Scalability Configuration:**  
     * **API Auto-scaling: Configure Horizontal Pod Autoscaler (HPA) for the API Deployment in Kubernetes, or equivalent auto-scaling settings in your cloud provider's service, based on CPU and/or memory utilization targets.**  
     * **Triton Tuning: Review Triton's dynamic batching settings (deployment/triton\_repo/vqa\_model/config.pbtxt) to optimize throughput and latency. Experiment with preferred batch sizes and max queue delay. Consider deploying multiple Triton instances behind its service if inference becomes a bottleneck.**  
     * **Redis Scaling: Ensure the managed Redis instance tier is appropriate for the expected load or configure Redis replication/clustering if self-hosting.**  
   * **Load Testing:**  
     * **Create a load\_tests/ directory.**  
     * **Write load testing scripts using a tool like k6 (JavaScript) or Locust (Python).**  
     * **Scripts should simulate realistic VQA requests (POST to /vqa with varying question lengths and potentially simulated image data sizes). Include pauses (think time).**  
     * **Run load tests against the staging or production environment (use caution with production). Start with low load and gradually increase.**  
     * **Monitor key metrics in Grafana during tests: API latency (p95, p99), API error rates (4xx, 5xx), CPU/memory usage (API pods, Triton pods), GPU utilization (Triton nodes), Redis performance, Triton inference latency/throughput.**  
   * **Performance Optimization:**  
     * **Analyze load test results and monitoring data to find bottlenecks.**  
     * **Implement optimizations:**  
       * **API: Profile Python code (cProfile), optimize slow endpoints, improve caching logic (Redis usage), check database query efficiency (if any). Use asynchronous code (async/await) effectively.**  
       * **Triton/Model: Consider model quantization (e.g., FP16) if latency is critical and accuracy impact is acceptable (requires model reconversion/re-export). Tune Triton configurations.**  
       * **Retrieval: Optimize vector database queries (indexing strategies in FAISS/ChromaDB).**  
     * **Re-run load tests to measure the impact of optimizations.**  
   * **Security Hardening:**  
     * **Secrets Management: Ensure all secrets (API keys, passwords, TLS certs) are stored securely using Kubernetes Secrets, Vault, or cloud provider secret managers, and injected into pods as environment variables or volumes, not baked into images.**  
     * **Network Policies: Implement Kubernetes Network Policies (or cloud security group rules) to restrict network traffic strictly (e.g., only allow API pods to talk to Triton and Redis, restrict public ingress).**  
     * **Dependency Scanning: Integrate a vulnerability scanner (e.g., pip-audit, trivy for containers, GitHub Dependabot alerts) into the CI pipeline or repository settings.**  
     * **Input Validation: Review API endpoint input validation (Pydantic models) to prevent injection attacks or unexpected inputs.**  
     * **Least Privilege: Ensure service accounts and cloud IAM roles used by applications have the minimum necessary permissions.**  
3. **Code Quality / Documentation: Ensure IaC and CI/CD code is clean, documented, and maintainable. Document the load testing methodology, results, and optimizations applied. Update security configurations and document security practices in the project's documentation.**

**Output Format: Provide code blocks for IaC configurations (if used), CI/CD pipeline definitions (\*.yml), example load testing scripts (\*.js for k6 or \*.py for Locust), key configuration snippets (e.g., HPA manifests, updated config.pbtxt, NetworkPolicy manifests). Summarize the key performance findings, optimizations implemented, and security measures added.**

---

**Prompt for Phase 14: Final Testing, Documentation & Handover**

**Role: You are the Lead Software Architect / QA Lead / Technical Writer responsible for final system validation and knowledge dissemination.**

**Goal: Ensure the VQA system is robust, thoroughly tested, and well-documented. Conduct final testing rounds including E2E and UAT, finalize all documentation for developers and operators, and prepare the project for handover or long-term maintenance.**

**Starting Codebase: Assume the complete project structure and implemented code from Phases 0 through 13 exists. The system is deployed, optimized, monitored, secured (basic hardening), and has CI/CD pipelines.**

**Primary References:**

* **Phased Implementation Plan (Phases 0-13)**  
* **"Next Steps" Summary**  
* **Existing Phase 0-13 Codebase (all modules, tests, deployment configs)**  
* **Test Results (Unit, Integration, Load from previous phases)**  
* **Architecture Document**  
* **User Feedback collected throughout the process**

**Instructions:**

1. **Target Files: Focus on creating new tests in tests/e2e/, significantly updating README.md, creating comprehensive documentation in a new docs/ directory, and potentially performing minor code refactoring across src/ for final polish.**  
2. **Implement Phase 14 Tasks:**  
   * **End-to-End (E2E) Testing:**  
     * **Create a new tests/e2e/ directory.**  
     * **Develop E2E test scripts that interact with the *deployed* system (or a dedicated E2E environment). Tools like pytest with requests (for API testing) or potentially UI automation frameworks (like Playwright or Selenium, if testing a web version of the AR experience becomes feasible) can be used.**  
     * **Test critical user flows:**  
       * **Submit VQA request (image \+ text) via API \-\> Verify plausible response structure/content.**  
       * **Submit feedback via API \-\> Verify it's stored correctly (requires test hooks or checking storage).**  
       * **Test interaction with RAG/Planner: Formulate questions likely to trigger retrieval or specific tools and verify behavior (might require analyzing logs or intermediate state if possible).**  
     * **Integrate E2E tests into the CD pipeline as smoke tests or run them periodically.**  
   * **User Acceptance Testing (UAT) (If Applicable):**  
     * **Define UAT scenarios covering key functionalities and target user goals (especially for the AR interface).**  
     * **Organize and conduct UAT sessions with representative users or stakeholders.**  
     * **Collect detailed feedback, classify bugs/issues, and prioritize necessary fixes.**  
     * **Implement critical fixes identified during UAT and verify them.**  
   * **Final Code Review and Polish:**  
     * **Conduct a final thorough review of the entire codebase (src/, scripts/, tests/, ar\_frontend/).**  
     * **Look for remaining TODOs, inconsistencies, potential bugs, unclear code, or areas lacking error handling.**  
     * **Refactor code for clarity, efficiency, and maintainability. Ensure adherence to PEP 8 / frontend best practices.**  
     * **Verify that all configurations (config.yaml, deployment manifests, CI/CD variables) are finalized and correct for the target environment(s). Remove any unused code or configuration.**  
   * **Documentation Finalization:**  
     * **README.md: Perform a major update to ensure it is the definitive entry point. Include:**  
       * **Clear Project Overview & Goals.**  
       * **Updated Feature List (mentioning all key components).**  
       * **Concise Quick Start / Setup instructions.**  
       * **Detailed instructions for Running API, Training, Evaluation, E2E Tests.**  
       * **Explanation of config.yaml structure.**  
       * **Overview of Deployment process (linking to detailed docs).**  
       * **AR Frontend setup/usage.**  
       * **High-level Project Structure map.**  
       * **Link to the docs/ directory for more details.**  
     * **docs/ Directory (Create comprehensive content):**  
       * **architecture.md: Detailed explanation of components (API, VLM, Triton, Planner, RAG, Connectors, Data Pipeline, AR Frontend, Monitoring) and their interactions, including sequence diagrams for key flows.**  
       * **api.md: Auto-generate (e.g., from FastAPI OpenAPI spec) or manually write detailed documentation for all API endpoints (request/response formats, authentication).**  
       * **developer\_guide.md: Environment setup (Python, Node.js, Docker), dependency management, running tests (unit, integration, e2e), coding style guidelines, branching strategy, CI/CD overview.**  
       * **operations\_guide.md: Deployment instructions (cloud provider specifics, IaC usage), monitoring setup (accessing Grafana, key dashboards/metrics), troubleshooting common issues, scaling procedures, backup/restore (if applicable).**  
       * **model\_management.md: How to train new models, evaluate them, export for Triton, update the deployed model version.**  
       * **data\_management.md: How to add/format new datasets, run synthetic data generation, process user feedback, manage active learning loop.**  
     * **Docstrings & Comments: Perform a final check of docstrings and comments throughout the codebase for accuracy, completeness, and clarity.**  
   * **Knowledge Transfer & Handover:**  
     * **Organize project code, documentation, and access credentials logically.**  
     * **Conduct walkthrough sessions with the team responsible for ongoing maintenance or operations.**  
     * **Ensure the CI/CD pipeline is stable and operational handover procedures are clear.**  
3. **Code Quality: Aim for production-ready code quality – robust, maintainable, well-tested, and secure. Documentation should be comprehensive, accurate, and easy for different audiences (developers, operators, users) to understand.**

**Output Format: Provide code blocks for E2E test examples (tests/e2e/test\_\*.py), the finalized README.mdcontent, and example structures/snippets for key documents within the docs/ directory. Summarize UAT findings and any significant final code refinements.**

