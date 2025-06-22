from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends, Request, status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String ,ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session,relationship
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
import cloudinary.uploader
import uuid
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Setup
DATABASE_URL = "postgresql://postgres:password@localhost:5432/fluttermusicapp"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"])
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String, unique=True)
    password = Column(String)

class Song(Base):
    __tablename__ = "songs"
    id = Column(Integer, primary_key=True)
    songid = Column(String)
    artist = Column(String)
    songname = Column(String)
    hex_code = Column(String)
    song_url = Column(String)
    thumbnail_url = Column(String)
    uploaded_by = Column(String)

class Favourite(Base):
    __tablename__ = "favourites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # FK to user table
    song_id = Column(Integer, ForeignKey("songs.id"))  # FK to song table

    user = relationship("User")
    song = relationship("Song")

Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI()

# Cloudinary
cloudinary.config( 
    cloud_name="dqyhuxexh", 
    api_key="", 
    api_secret="", 
    secure=True
)

# Schemas
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    message: str

class FavouriteCreate(BaseModel):
    user_id: int
    song_id: int
# Auth utils
def create_access_token(data: dict):
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    token = auth.split()[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return {"user_email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# Auth Routes
@app.post("/signup/", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"username": user.username, "email": user.email, "message": "User created"}

@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {
        "message": "Login successful",
        "token": token,
        "username": db_user.username,
        "user_id": db_user.id  # ✅ THIS MUST BE INCLUDED
    }


# Upload song
@app.post("/upload/")
def upload_song(
    song: UploadFile = File(...),
    thumbnail: UploadFile = File(...),
    artist: str = Form(...),
    songname: str = Form(...),
    hex_code: str = Form(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    songid = str(uuid.uuid4())
    try:
        song_res = cloudinary.uploader.upload(song.file, resource_type='auto', folder=f"songs/{songid}")
        thumb_res = cloudinary.uploader.upload(thumbnail.file, resource_type='image', folder=f"songs/{songid}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    new_song = Song(
        songid=songid,
        artist=artist,
        songname=songname,
        hex_code=hex_code,
        song_url=song_res['secure_url'],
        thumbnail_url=thumb_res['secure_url'],
        uploaded_by=current_user["user_email"]
    )
    db.add(new_song)
    db.commit()
    return {"message": "Uploaded successfully"}

# List songs
@app.get("/list")
def list_songs(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    return db.query(Song).filter(Song.uploaded_by == current_user["user_email"]).all()

@app.get("/all_songs")
def get_all_songs(db: Session = Depends(get_db)):
    return db.query(Song).order_by(Song.id.desc()).all()

@app.post("/favourite", tags=["Favourites"])
def add_favourite(fav: FavouriteCreate, db: Session = Depends(get_db)):
    exists = db.query(Favourite).filter_by(user_id=fav.user_id, song_id=fav.song_id).first()
    if exists:
        raise HTTPException(status_code=400, detail="Already favorited")

    db_fav = Favourite(user_id=fav.user_id, song_id=fav.song_id)
    db.add(db_fav)
    db.commit()
    db.refresh(db_fav)
    return {"message": "Added to favourites"}

@app.get("/favourite/{user_id}", tags=["Favourites"])
def get_user_favourites(user_id: int, db: Session = Depends(get_db)):
    favourites = (
        db.query(Song)
        .join(Favourite, Song.id == Favourite.song_id)
        .filter(Favourite.user_id == user_id)
        .distinct()
        .all()
    )
    return favourites

@app.delete("/favourite", tags=["Favourites"])
def remove_favourite(fav: FavouriteCreate, db: Session = Depends(get_db)):
    existing = db.query(Favourite).filter_by(user_id=fav.user_id, song_id=fav.song_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(existing)
    db.commit()
    return {"message": "Removed from favourites"}

@app.get("/favourite_ids/{user_id}")
def get_fav_ids(user_id: int, db: Session = Depends(get_db)):
    return [fav.song_id for fav in db.query(Favourite).filter_by(user_id=user_id).all()]

@app.get("/recommend/{user_id}", tags=["Recommendations"])
def recommend_songs(user_id: int, db: Session = Depends(get_db)):
    # Fetch all songs and user favorites
    all_songs = db.query(Song).all()
    user_favs = db.query(Song).join(Favourite).filter(Favourite.user_id == user_id).all()

    if not user_favs:
        raise HTTPException(status_code=404, detail="User has no favourites yet.")

    # Build DataFrame
    data = [{
        'id': song.id,
        'songname': song.songname,
        'artist': song.artist,
        'genre': song.hex_code,  # You can change this if you store genre
    } for song in all_songs]

    df = pd.DataFrame(data)

    # Combine songname + artist + genre to create features
    df['features'] = df['songname'] + " " + df['artist'] + " " + df['genre']

    # Vectorize features
    vectorizer = CountVectorizer().fit_transform(df['features'])
    similarity_matrix = cosine_similarity(vectorizer)

    # Get indices of favorite songs
    fav_ids = [song.id for song in user_favs]
    fav_indices = df[df['id'].isin(fav_ids)].index.tolist()

    # Get similarity scores for all songs
    scores = similarity_matrix[fav_indices].mean(axis=0)
    df['score'] = scores

    # Recommend top 5 songs not already in favorites
    recommendations = df[~df['id'].isin(fav_ids)].sort_values(by='score', ascending=False).head(5)

    # Fetch original song records
    recommended_songs = db.query(Song).filter(Song.id.in_(recommendations['id'].tolist())).all()

    return recommended_songs
