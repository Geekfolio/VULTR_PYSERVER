import os
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2
from typing import Optional, Dict
import logging
import subprocess
from docx import Document
from odf import text, teletype
from odf.opendocument import load
import textract
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import requests
import json
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Access the variables
vultr_infer = os.getenv("VULTR_INFER")


# Get the API key from environment variable
api_key = vultr_infer

# Set the API endpoint and model
api_base = "https://api.vultrinference.com/v1/chat/completions"
model = 'llama2-13b-chat-Q5_K_M' #"Nous-Hermes-2-Mixtral-8x7B-DPO-Q5_K_M" 


# Prepare the headers and payload
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# FastAPI app initialization
app = FastAPI()

class ResumeExtractor:
    def __init__(self, upload_dir: str = "./uploaded_files"):
        """Initialize the ResumeExtractor with upload directory."""
        self.upload_dir = upload_dir
        self._ensure_upload_dir()
        self.supported_formats = {
            'pdf': self.extract_text_from_pdf,
            'image': self.extract_text_from_image,
            'docx': self.extract_text_from_docx,
            'doc': self.extract_text_from_doc,
            'odt': self.extract_text_from_odt
        }

    def _ensure_upload_dir(self) -> None:
        """Create upload directory if it doesn't exist."""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    @staticmethod
    def check_file_type(file_path: str) -> str:
        """Check file extension and return file type."""
        ext = os.path.splitext(file_path)[-1].lower()
        format_mapping = {
            '.pdf': 'pdf',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.docx': 'docx',
            '.doc': 'doc',
            '.odt': 'odt'
        }
        if ext in format_mapping:
            return format_mapping[ext]
        raise ValueError(f"Unsupported file type: {ext}")

    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text using appropriate method based on file type."""
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_type}")

        return self.supported_formats[file_type](file_path)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods."""
        methods = [
            (self._extract_with_pdfplumber, "pdfplumber"),
            (self._extract_with_pypdf2, "PyPDF2"),
            (self._extract_with_ocr, "OCR")
        ]

        for extract_method, method_name in methods:
            try:
                text = extract_method(file_path)
                if text.strip():
                    logger.info(f"Successfully extracted text using {method_name}")
                    return text
                logger.info(f"No text found using {method_name}, trying next method...")
            except Exception as e:
                logger.error(f"Error with {method_name}: {str(e)}")

        return ""

    @staticmethod
    def _extract_with_pdfplumber(file_path: str) -> str:
        """Extract text using pdfplumber."""
        with pdfplumber.open(file_path) as pdf:
            return ' '.join(page.extract_text() or '' for page in pdf.pages)

    @staticmethod
    def _extract_with_pypdf2(file_path: str) -> str:
        """Extract text using PyPDF2."""
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            return ' '.join(page.extract_text() or '' for page in reader.pages)

    @staticmethod
    def _extract_with_ocr(file_path: str) -> str:
        """Extract text using OCR."""
        images = convert_from_path(file_path)
        return ' '.join(pytesseract.image_to_string(image) for image in images)

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using pytesseract."""
        try:
            with Image.open(file_path) as image:
                return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            # Try using mammoth first for better formatting preservation
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
    
            if text.strip():
                return text
    
            # Fallback to python-docx if mammoth fails
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return ""

    def extract_text_from_doc(self, file_path: str) -> str:
        """Extract text from DOC file."""
        try:
            # Try textract first
            return textract.process(file_path).decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting text from DOC with textract: {str(e)}")
            try:
                # Fallback to antiword if available
                return subprocess.check_output(['antiword', file_path]).decode('utf-8')
            except Exception as e2:
                logger.error(f"Antiword fallback failed: {str(e2)}")
                return ""

    def extract_text_from_odt(self, file_path: str) -> str:
        """Extract text from ODT file."""
        try:
            textdoc = load(file_path)
            allparas = textdoc.getElementsByType(text.P)
            return '\n'.join([teletype.extractText(para) for para in allparas])
        except Exception as e:
            logger.error(f"Error extracting text from ODT: {str(e)}")
            # Fallback to textract
            try:
                return textract.process(file_path).decode('utf-8')
            except Exception as e2:
                logger.error(f"Textract fallback failed: {str(e2)}")
                return ""

def clean_json_string(json_str):
    # Remove any leading or trailing whitespace
    json_str = json_str.strip()
    json_str = re.sub(r'`', '', json_str)
    json_str = re.sub(r'json', '', json_str)

    # # Ensure the string starts with { and ends with }
    # if not json_str.startswith('{'):
    #     json_str = '{' + json_str
    # if not json_str.endswith('}'):
    #     json_str = json_str + '}'

    # # Replace any single quotes with double quotes
    # json_str = json_str.replace("'", '"')

    # Fix common formatting issues
    # json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
    # json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas


    return json_str




def generate_json_from_text(text):
    prompt = f'''
    I will provide you with a resume. The content might be a little jumbled. Can you analyze it and convert it into the following structured JSON format?

    Only output valid JSON. Do not include any extra commentary or text, just the JSON data. Leave any missing fields empty as shown in the format.

    The resume content is:
    {text}

    Format the content in the resume to this JSON: {{
    "personal_information": [
        {{
        "name": [""],
        "contact_information": [
            {{
            "phone_number": [""],
            "email": [""],
            "address": [""]
            }}
        ],
        "linkedin_profile": [""],
        "github_profile": [""],
        "objective_summary": [
            {{
            "career_objective": [""],
            "professional_summary": [""]
            }}
        ]
        }}
    ],
    "education": [
        {{
        "degree": [""],
        "major_field_of_study": [""],
        "university_institution_name": [""],
        "graduation_date": [""],
        "cgpa_grades": [""]
        }}
    ],
    "experience": [
        {{
        "job_title": [""],
        "company_name": [""],
        "location": {{
            "city": [""],
            "state": [""]
        }},
        "dates_of_employment": {{
            "start_date": [""],
            "end_date": [""]
        }},
        "responsibilities_achievements": [""]
        }}
    ],
    "projects": [
        {{
        "project_title": [""],
        "technologies_used": [""],
        "project_description": [""],
        "duration": {{
            "start_date": [""],
            "end_date": [""]
        }},
        "project_links": [""]
        }}
    ],
    "certifications": [
        {{
        "certification_title": [""],
        "issuing_organization": [""],
        "date_obtained": [""]
        }}
    ],
    "skills": {{
        "technical_skills": [""],
        "soft_skills": [""]
    }},
    "achievements": {{
        "awards_honors": [""],
        "scholarships": [""],
        "competitions": [""]
    }},
    "extracurricular_activities": {{
        "clubs_organizations": [""],
        "volunteer_work": [""],
        "leadership_roles": [""]
    }},
    "languages": [
        {{
        "language_proficiency": [""],
        "level_of_proficiency": [""]
        }}
    ]
    }}'''

    try:
        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': 2000
        }

        # Make the request
        response = requests.post(api_base, headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            response_data = response.json()
            json_str = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            logger.error(f"vultr call Error: {response.status_code}, {response.text}")
            return {"error": "Failed to generate JSON from text."}

        # Clean and fix the JSON string
        cleaned_json_str = clean_json_string(json_str)
        
        # Parse and format the JSON
        try:            
            structured_json=cleaned_json_str
            #structured_json = parse_model_output(structured_json)
            return structured_json
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON after cleaning: {str(e)}")
            return {"Warning": "Not all data fetchable", "raw_text": cleaned_json_str}
        
    except Exception as e:
        logger.error(f"Error generating JSON: {str(e)}")
        return {"error": str(e)}
 



def parse_model_output(model_output: str) -> dict:
    # Initialize the JSON structure
    json_structure = {
        "personal_information": [
            {
                "name": [""],
                "contact_information": [
                    {
                        "phone_number": [""],
                        "email": [""],
                        "address": [""]
                    }
                ],
                "linkedin_profile": [""],
                "github_profile": [""],
                "objective_summary": [
                    {
                        "career_objective": [""],
                        "professional_summary": [""]
                    }
                ]
            }
        ],
        "education": [
            {
                "degree": [""],
                "major_field_of_study": [""],
                "university_institution_name": [""],
                "graduation_date": [""],
                "cgpa_grades": [""]
            }
        ],
        "experience": [
            {
                "job_title": [""],
                "company_name": [""],
                "location": {
                    "city": [""],
                    "state": [""]
                },
                "dates_of_employment": {
                    "start_date": [""],
                    "end_date": [""]
                },
                "responsibilities_achievements": [""]
            }
        ],
        "projects": [
            {
                "project_title": [""],
                "technologies_used": [""],
                "project_description": [""],
                "duration": {
                    "start_date": [""],
                    "end_date": [""]
                },
                "project_links": [""]
            }
        ],
        "certifications": [
            {
                "certification_title": [""],
                "issuing_organization": [""],
                "date_obtained": [""]
            }
        ],
        "skills": {
            "technical_skills": [""],
            "soft_skills": [""]
        },
        "achievements": {
            "awards_honors": [""],
            "scholarships": [""],
            "competitions": [""]
        },
        "extracurricular_activities": {
            "clubs_organizations": [""],
            "volunteer_work": [""],
            "leadership_roles": [""]
        },
        "languages": [
            {
                "language_proficiency": [""],
                "level_of_proficiency": [""]
            }
        ]
    }
    # Example regex patterns for extracting data
    patterns = {
        "name": r"(?i)name\s*:\s*(.+)",
        "phone": r"(?i)phone\s*number\s*:\s*(.+)",
        "email": r"(?i)email\s*:\s*(.+)",
        "address": r"(?i)address\s*:\s*(.+)",
        "linkedin": r"(?i)linkedin\s*profile\s*:\s*(.+)",
        "github": r"(?i)github\s*profile\s*:\s*(.+)",
        "career_objective": r"(?i)career\s*objective\s*:\s*(.+)",
        "professional_summary": r"(?i)professional\s*summary\s*:\s*(.+)",
        "degree": r"(?i)degree\s*:\s*(.+)",
        "major": r"(?i)major\s*field\s*of\s*study\s*:\s*(.+)",
        "university": r"(?i)university\s*institution\s*name\s*:\s*(.+)",
        "graduation_date": r"(?i)graduation\s*date\s*:\s*(.+)",
        "cgpa": r"(?i)cgpa\s*grades\s*:\s*(.+)",
        "job_title": r"(?i)job\s*title\s*:\s*(.+)",
        "company_name": r"(?i)company\s*name\s*:\s*(.+)",
        "city": r"(?i)city\s*:\s*(.+)",
        "state": r"(?i)state\s*:\s*(.+)",
        "start_date": r"(?i)start\s*date\s*:\s*(.+)",
        "end_date": r"(?i)end\s*date\s*:\s*(.+)",
        "responsibilities": r"(?i)responsibilities\s*achievements\s*:\s*(.+)",
        "project_title": r"(?i)project\s*title\s*:\s*(.+)",
        "technologies_used": r"(?i)technologies\s*used\s*:\s*(.+)",
        "project_description": r"(?i)project\s*description\s*:\s*(.+)",
        "project_links": r"(?i)project\s*links\s*:\s*(.+)",
        "certification_title": r"(?i)certification\s*title\s*:\s*(.+)",
        "issuing_organization": r"(?i)issuing\s*organization\s*:\s*(.+)",
        "date_obtained": r"(?i)date\s*obtained\s*:\s*(.+)",
        "technical_skills": r"(?i)technical\s*skills\s*:\s*(.+)",
        "soft_skills": r"(?i)soft\s*skills\s*:\s*(.+)",
        "awards_honors": r"(?i)awards\s*honors\s*:\s*(.+)",
        "scholarships": r"(?i)scholarships\s*:\s*(.+)",
        "competitions": r"(?i)competitions\s*:\s*(.+)",
        "clubs": r"(?i)clubs\s*organizations\s*:\s*(.+)",
        "volunteer_work": r"(?i)volunteer\s*work\s*:\s*(.+)",
        "leadership_roles": r"(?i)leadership\s*roles\s*:\s*(.+)",
        "language_proficiency": r"(?i)language\s*proficiency\s*:\s*(.+)",
        "level_of_proficiency": r"(?i)level\s*of\s*proficiency\s*:\s*(.+)"
    }

    # Extract data using regex patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, model_output)
        if match:
            value = match.group(1).strip()
            if key in ["project_title", "technologies_used", "project_description", "project_links"]:
                # Append project details if the project structure is already there
                if len(json_structure["projects"]) == 0 or json_structure["projects"][-1].get("project_title") != [""]:
                    json_structure["projects"].append({
                        "project_title": [""],
                        "technologies_used": [""],
                        "project_description": [""],
                        "duration": {
                            "start_date": [""],
                            "end_date": [""]
                        },
                        "project_links": [""]
                    })
                if json_structure["projects"]:
                    current_project = json_structure["projects"][-1]
                    current_project[key] = [value]
            elif key in ["certification_title", "issuing_organization", "date_obtained"]:
                # Append certification details if the certification structure is already there
                if len(json_structure["certifications"]) == 0 or json_structure["certifications"][-1].get("certification_title") != [""]:
                    json_structure["certifications"].append({
                        "certification_title": [""],
                        "issuing_organization": [""],
                        "date_obtained": [""]
                    })
                if json_structure["certifications"]:
                    current_certification = json_structure["certifications"][-1]
                    current_certification[key] = [value]
            elif key in json_structure["skills"]:
                json_structure["skills"][key] = [value]
            elif key in json_structure["achievements"]:
                json_structure["achievements"][key] = [value]
            elif key in json_structure["extracurricular_activities"]:
                json_structure["extracurricular_activities"][key] = [value]
            elif key in json_structure["languages"]:
                # Assume language entries are lists; append directly
                json_structure["languages"].append({
                    "language_proficiency": [value],
                    "level_of_proficiency": [""]
                })
            else:
                # For other keys, set directly in the JSON structure
                if "." in key:
                    # Handle nested keys
                    keys = key.split(".")
                    current_level = json_structure
                    for subkey in keys[:-1]:
                        current_level = current_level.setdefault(subkey, {})
                    current_level[keys[-1]] = [value]
                else:
                    json_structure[key] = [value]

    return json_structure

@app.get("/")
async def read_root(): 
    return {"Hello": "World"}

@app.get("/extract")
async def extract_text_from_resume():
    return {"message": "Upload a resume file to extract text."}


@app.post("/extract")
async def extract_text_from_resume(file: UploadFile = File(...)):
    extractor = ResumeExtractor()

    try:
        file_location = f"{extractor.upload_dir}/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Identify file type
        file_type = extractor.check_file_type(file_location)

        # Extract text from file
        extracted_text = extractor.extract_text(file_location, file_type)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from the resume.")

        # Generate structured JSON from the extracted text
        structured_json = generate_json_from_text(extracted_text)
        return JSONResponse(content={"extracted_text": extracted_text, "structured_json": structured_json})

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "_main_":
    uvicorn.run("vapp_llama:app", host="0.0.0.0", port=8000, reload=True)
