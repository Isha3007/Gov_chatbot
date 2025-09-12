cd server

python ingest.py --reset --websites-file websites.txt

uvicorn server:app --reload --port 8000

cd web

npm run dev
