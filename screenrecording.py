import numpy as np 
import cv2
from PIL import ImageGrab
import pytesseract
import os
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from pynput import keyboard
from pynput import keyboard
import matplotlib.pyplot as plt
import pyautogui
import re
from pynput import mouse
import requests
from fuzzywuzzy import process

load_dotenv()

client = OpenAI()

# Global variables
click = 0
selection_points = []
capture_mode = False


question_data = {
    'Which of the following will mark the furthest extent of a continent?': 'D. The lowest part of the continental slope',
    'Which of the following locations is an example of spontaneous subduction?': 'A. Mariana Trench',
    'Which statement below about subduction angle is correct?': 'B. Old, cold slabs subduct at steep angles',
    'The following is an example of a narrow linear sea during ocean basin formation.': 'D. The Red Sea',
    'Which name applied to the processes that collectively produce mountain belts?': 'A. orogenesis',
    'At what type of plate boundary would Andean-type subduction zones occur?': 'B. oceanic-continental convergent',
    'When an inactive island arc enters a subduction zone, a terrane is added to the edge of the continent. How can this be explained?': 'C. The rocks in the island arc are too buoyant to subduct',
    "Which portion of North America was 'stretched' during the past 20 million years?": 'D. Basin and Range province',
    "What term refers to the fact that the crust 'floats' in gravitational balance on a denser mantle?": 'C. Isostasy',
    'The ________ is an elevational point that divides an entire continent into large drainage basins.': 'B. Continental Divide',
    "________ is a measure of a stream's ability to transport particles based on size rather than quantity.": 'C. Competence',
    '________ occurs when a rock is gradually dissolved by flowing water.': 'B. Corrosion',
    'As stream velocity slows, the largest particles get deposited first.': 'True',
    'Point bars are locations where erosion occurs whereas cutbanks are locations where deposition occurs.': 'False',
    'What is base level?': 'B. The downward limit of erosion',
    'The upper limit of the zone of saturation is called the __________.': 'B. water table',
    'What are permeable layers of rock, such as layers of sand or gravel that freely transmit groundwater, called?': 'C. aquifers',
    'What mathematical relationship relates groundwater velocity, gradient, viscosity of the fluid, and permeability of the aquifer?': "B. Darcy's law",
    'Where are most caverns created?': 'D. at or just below the water table',
    'What is the age of a rock sample if the ratio of radioactive parent to stable daughter product is 1:1 and the half-life is 5 million years?': 'C. 5 million years',
    'What is the age of a rock sample containing the same hypothetical isotope as the previous one if the parent-daughter ratio is about 1:7?': 'C. 15 million years',
    'According to Figure 9.25, which statement about the age of Mancos shale is most likely correct?': 'B. It is older than 66 million years but younger than 160 million years',
    'In the Grand Canyon, the rock layers on the top should be _______ than those at the bottom of the canyon.': 'B. younger',
    'If a fault does not cut across a rock layer at shallower depth, the fault is _______ than the rock layer in age.': 'B. older',
    'If a rock layer contains some particles from the other rock layer, then the rock layer is _____ than the other layer.': 'A. younger',
    'At the end of Precambrian, the current continents of South America, Africa, Australia, Antarctica, India comprised the vast southern continent of ________.': 'B. Gondwana',
    'Abundant fossil evidence did not appear in the geologic record until about ______.': 'A. 550 million years ago',
    'During the _______ era, the westward-moving North American plate began to override the Pacific plate, eventually causing the tectonic activity that ultimately formed the mountains of western North America.': 'C. Mesozoic',
    'Oil and Gas window': 'B. 100-250C',
    "The greenhouse gases absorb ___________ wavelength radiations emitted from Earth's surface.": 'B. long',
    'Natural causes for climate change include ____________.': 'D. all of above'
}




def find_dataset_answer(question):
    global question_data

    best_match, score = process.extractOne(question, question_data.keys())

    if score > 80:
        return best_match + ": " + question_data[best_match]
    else:
        return None

def perform_google_search(query):
    """Perform a Google search using Custom Search API and return the response."""
    print("Performing Google Search for query:", query)
    api_key = os.getenv('GOOGLE_SEARCH_API')
    cx = os.getenv('GOOGLE_SEARCH_CX')

    search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
    response = requests.get(search_url)

    if response.status_code == 200:
        response_json = response.json()
        search_results = ''

        if "items" in response_json:
            for item in response_json["items"]:
                search_results += f'Title: {item["title"]}, Snippet: {item["snippet"]}\n'
        else:
            print("No search results found.")
        return search_results
    else:
        print(f"Google Search API Error: {response.status_code}")
        return ""



def extract_question_data(image):
    """Extract OCR data and parse the question."""

    os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l box3 --oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    question_text = pytesseract.image_to_string(image, config=config)

    question_filtered = ''
    parsing_active = False

    for i, word in enumerate(ocr_data['text']):
        if word in {'O', '@', '©', '®'}:
            break

        if "pts" in word:
            parsing_active = True
            continue

        if parsing_active:
            question_filtered += word + " "

    return ocr_data, question_text, question_filtered.strip()

def execute_gpt_response(question, context):
    """Send the question and Google search results to GPT-4 and retrieve a response."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Context:\n" + context + """
                    Respond to the question bellow this paragraph using the context above. The context could
                    include Google search results, or just the answer to the question.
                    Respond with the number of the correct answer only, starting from 1 for the 
                    first choice, 2 for the second, and so on. Do not include any explanation 
                    or additional text. If there is math use python to solve it.\n Question: """ + question
                }
            ]
        )

        gpt_response = completion.choices[0].message.content

        if gpt_response.isnumeric():
            return gpt_response
        else:
            print("Invalid response from GPT-4:", gpt_response)
            return None
           
    except Exception as e:
        print("GPT API Error:", e)
        return None

def move_mouse(data, offsetx, offsety, answer_index):
    """Move the mouse pointer to the corresponding answer location."""
    answer_coords = []
    NextCoordinates = []

    for i, word in enumerate(data['text']):
        x = data['left'][i] + offsetx
        y = data['top'][i] + offsety

        if word == 'O':
            answer_coords.append((x, y))

        if word == '@' or word == '©' or word == '@©' or word == '©@' or word == "®":
        
            answer_coords.append((x, y))
       
        if word == 'Next':
            NextCoordinates.append((x, y))
           
    try:
        if answer_coords:
            target = answer_coords[int(answer_index) - 1]
            pyautogui.moveTo(target[0], target[1], duration=0.5)
    except Exception as e:
        print(f"Error moving to answer: {e}")



def capture_screen_and_process(offsetx = 329, offsety = 294, x2 = 971, y2 = 588):
    # Capture the full screen
    if x2 < offsetx or y2 < offsety:
        print("Invalid coordinates")
        return
    
    screen = (ImageGrab.grab(bbox=(offsetx, offsety, x2, y2)))

    screen = np.array(screen)

    data, question, question_only = extract_question_data(screen)

    if question_only is None:
        print("No question found.")
        return

    print(question_only)
    
    google_data = perform_google_search(question_only)
 

    # Use either the dataset response or the Google response (whichever is available)
    context =  google_data


    response = execute_gpt_response(question, context)

    if response:    
        move_mouse(data, offsetx, offsety, response)

def on_click(x, y, button, pressed):
    """Handle mouse clicks for screen selection."""
    global capture_mode, selection_points, click 

    if capture_mode and pressed and button == mouse.Button.left:
        selection_points.append((x, y))
        if len(selection_points) == 2:
            x1, y1 = selection_points[0]
            x2, y2 = selection_points[1]
            capture_screen_and_process(x1, y1, x2, y2)
            capture_mode = False
            selection_points.clear()


    else:
        try:
            if pressed and button == mouse.Button.left:
                click += 1
                if click % 2 == 1:
                    print(f"Right-click at ({x}, {y})")
                    capture_screen_and_process()
        except AttributeError:
            # Handle special keys that don't have a char attribute
            pass

def on_key_press(key):
    """Handle keyboard input."""
    global capture_mode

    try:
        if key.char == 'p':
            print("Enter screen selection mode.")
            capture_mode = True

        if key.char == 'r':
            print("Capture full screen.")
            capture_screen_and_process()
    except AttributeError:
        pass

# Start the keyboard listener
with mouse.Listener(on_click=on_click) as mouse_listener, keyboard.Listener(on_press=on_key_press) as keyboard_listener:
    print("Listening for 'p' key press to initiate point capture.")
    mouse_listener.join()
    keyboard_listener.join()