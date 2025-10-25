# Manslater: The Translator You Didn’t Know You Needed

You might have seen the Oppenheimer meme: “Bro was one of the smartest men alive but still couldn’t understand a woman.” Well, I bought a solution to fix that. Introducing Manslater: it translates what women say into what they actually mean. 😅

This project isn't just running off-the-shelf magic; it's powered by a custom fine-tuned model based on OpenAI’s GPT-3.5 Turbo! The biggest challenge? Crafting the perfect dataset. I had to write, rewrite, and validate a custom dataset multiple times to achieve those uncannily accurate (and often hilarious) translations.

## Working(SS):
<img width="943" height="671" alt="Screenshot 2025-10-21 at 8 42 08 PM" src="https://github.com/user-attachments/assets/8c7c02e7-5a7e-4367-9201-76fc2a2a9afd" />
<img width="943" height="671" alt="Screenshot 2025-10-21 at 8 41 37 PM" src="https://github.com/user-attachments/assets/36bc1910-ae26-48f8-a407-3a986412b319" />



## Tech Stack

### Frontend
- **React.js**: A JavaScript library for building user interfaces.
- **HTML2Canvas**: A library to take screenshots of the browser.
- **Vercel Analytics**: For web analytics.

### Backend
- **Flask**: A Python web framework.
- **Gunicorn**: A Python WSGI HTTP Server for UNIX.
- **Flask-Cors**: A Flask extension for handling Cross Origin Resource Sharing (CORS).
- **python-dotenv**: For loading environment variables from a `.env` file.
- **requests**: Python HTTP library.

## How to Run Locally

To run Manslater locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Manslater.git
cd Manslater
```

### 2. Backend Setup

Navigate to the `manslator-backend` directory:

```bash
cd manslator-backend
```

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts/activate`
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the `manslator-backend` directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Run the Flask backend:

```bash
python app.py
```

The backend server will typically run on `http://127.0.0.1:5000`.

### 3. Frontend Setup

Open a new terminal and navigate to the `manslator-frontend` directory:

```bash
cd manslator-frontend
```

Install the Node.js dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm start
```

The frontend application will typically open in your browser at `http://localhost:3000`.

Make sure both the backend and frontend servers are running simultaneously to use the application.

What are you looking for? The dataset and code that I used to fine tune, right. Thats not public. HA HA HA.
