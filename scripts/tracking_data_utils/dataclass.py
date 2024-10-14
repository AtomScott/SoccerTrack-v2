from typing import List, Optional
from pydantic import BaseModel, Field, validator, root_validator, AliasChoices

class Player(BaseModel):
    id: int = Field(validation_alias=AliasChoices('@playerId', '@id'))
    loc: List[float] = Field(None, alias='@loc', description="Location of the player on the pitch as [x, y].")
    speed: float = Field(None, alias='@speed', description="Speed of the player.")
    name: Optional[str] = Field(None, alias='@name')
    nameEn: Optional[str] = Field(None, alias='@nameEn')
    shirtNumber: Optional[int] = Field(None, alias='@shirtNumber')
    position: Optional[str] = Field(None, alias='@position')
    teamId: Optional[int] = Field(None, alias='@teamId')
    teamName: Optional[str] = Field(None, alias='@teamName')
    teamNameEn: Optional[str] = Field(None, alias='@teamNameEn')

    @validator('loc', pre=True)
    def parse_loc(cls, v):
        if isinstance(v, str):
            try:
                return eval(v)
            except:
                raise ValueError(f"Could not parse @loc: {v}")
        return v

    class Config:
        populate_by_name = True

class Ball(BaseModel):
    loc: List[float] = Field(..., alias='@loc', description="Location of the ball on the pitch as [x, y].")
    speed: Optional[float] = Field(None, alias='@speed', description="Speed of the ball.")  # Now optional to handle 'NA'

    @validator('speed', pre=True)
    def parse_speed(cls, v):
        if v == 'NA':
            return None  

    @validator('loc', pre=True)
    def parse_loc(cls, v):
        if isinstance(v, str):
            try:
                return eval(v)
            except:
                raise ValueError(f"Could not parse @loc: {v}")
        return v

class Chunk(BaseModel):
    period: str = Field(..., alias='@period')
    matchTime: int = Field(..., alias='@matchTime')
    players: List[Player] = Field(..., alias='player')
    

class Team(BaseModel):
    id: int = Field(..., alias='@id')
    name: str = Field(..., alias='@name')
    nameEn: str = Field(..., alias='@nameEn')
    side: str = Field(..., alias='@side')

class Pitch(BaseModel):
    width: int = Field(..., alias='@width')
    height: int = Field(..., alias='@height')

class Period(BaseModel):
    period: str = Field(..., alias='@period')
    frameStart: Optional[float] = Field(None, alias='@frameStart')
    frameEnd: Optional[float] = Field(None, alias='@frameEnd')
    matchTimeStart: Optional[float] = Field(None, alias='@matchTimeStart')
    matchTimeEnd: Optional[float] = Field(None, alias='@matchTimeEnd')
    fps: int = Field(..., alias='@fps')

    @validator('*', pre=True)
    def convert_nan_to_none(cls, v):
        if v == 'nan':
            return None
        return v

class Match(BaseModel):
    matchId: int = Field(..., alias='@matchId')
    matchTitle: str = Field(..., alias='@matchTitle')
    matchDatetime: str = Field(..., alias='@matchDatetime')
    matchDatetimeLocal: str = Field(..., alias='@matchDatetimeLocal')
    timezone: str = Field(..., alias='@timezone')
    matchFullTime: int = Field(..., alias='@matchFullTime')
    matchExtraTime: int = Field(..., alias='@matchExtraTime')
    periods: List[Period] = []

    @root_validator(pre=True)
    def parse_periods(cls, values):
        # Assuming the XML to dict conversion wraps periods in a 'period' key
        periods_data = values.get('period', [])
        if isinstance(periods_data, dict):
            # If there's only one period, it would be parsed as a dict, not a list
            periods_data = [periods_data]
        values['period'] = periods_data
        return values


class MatchMetadata(BaseModel):
    match: Match
    pitch: Pitch
    teams: List[Team]
    players: List[Player]
    activePlayers: List[Chunk]

    class Config:
        populate_by_name = True

    @root_validator(pre=True)
    def parse_teams(cls, values):
        # Assuming the XML to dict conversion wraps periods in a 'period' key
        teams_data = values.get('teams', [])

        if isinstance(teams_data, dict):
            teams_data = teams_data.get('team', [])
        
        values['teams'] = teams_data
        return values

    @root_validator(pre=True)
    def parse_players(cls, values):
        players_data = values.get('players', [])

        if isinstance(players_data, dict):
            players_data = players_data.get('player', [])
        values['players'] = players_data
        return values
    
    @root_validator(pre=True)
    def parse_active_players(cls, values):
        active_player_data = values.get('activePlayers', [])
        if isinstance(active_player_data, dict):
            active_player_data = active_player_data.get('chunk', [])
        values['activePlayers'] = active_player_data
        return values


    
class Frame(BaseModel):
    matchTime: int = Field(..., alias='@matchTime', description="Match time in milliseconds.")
    frameNumber: int = Field(..., alias='@frameNumber', description="Frame number.")
    eventPeriod: str = Field(..., alias='@eventPeriod', description="Period of the event (e.g., FIRST_HALF, SECOND_HALF).")
    ballStatus: str = Field(..., alias='@ballStatus', description="Status of the ball (e.g., IN_PLAY, OUT_OF_PLAY).")
    players: List[Player] = Field(..., alias='player', description="List of player instances in the frame.")
    ball: Ball = Field(..., description="Ball instance in the frame.")

    class Config:
        populate_by_name = True