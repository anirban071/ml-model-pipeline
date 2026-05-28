# Use a lightweight Python image
FROM python:3.11.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install them first (takes advantage of Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (app.py and model.pkl)
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]