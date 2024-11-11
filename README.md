# PyServer

# Resume Extraction API

This API server allows you to upload resumes in various formats (PDF, DOCX, ODT, PNG) and returns structured data extracted from these documents. It uses Uvicorn and FastAPI to serve the API, and requires OCR and other processing tools.

## Requirements

This project requires a few system packages for text extraction and OCR processing. Run the following command to install them:


### Dependencies:

      - tesseract
      - poppler-utils
      - swig
      - libpulse

```bash
sudo apt install tesseract-ocr poppler-utils swig libpulse-dev
```

### Setup

This project uses Conda to manage dependencies. Make sure you have Miniconda or Anaconda installed. Then, set up the environment with the following steps:

### Clone the repository:

```bash
git clone https://github.com/Geekfolio/PyServer
cd PyServer
```

### Create and activate the environment from the environment.yaml file:

```bash
conda env create -f environment.yaml
conda activate vult
```

### Add .env file 
To execute this, create a .env file add your API key as 
```
VULTR_INFER=<your key>
```


### Running the API Server

To start the server in development mode, use the following command:
```bash
uvicorn server:app --reload
```

This will start the server at http://127.0.0.1:8000.

### Usage

You can send POST requests to upload a resume file to the endpoint /upload. The server will process the file and return structured data in JSON format.
Example Request

```bash
curl -X POST "http://127.0.0.1:8000/extract" -F "file=@path_to_resume"
```
Replace path_to_resume with the path to your resume file.
