import pandas as pd
from preprocessing import Event_data

event_path = "/home/atom/SoccerTrack-v2/data/raw/117093/117093_player_nodes.csv"
tracking_path = "/home/atom/SoccerTrack-v2/data/raw/117093/117093_tracker_box_data.xml"
meta_path = "/home/atom/SoccerTrack-v2/data/raw/117093/117093_tracker_box_metadata.xml"

# Load and process soccer data
soccertrack_df = Event_data("bepro", event_path, st_track_path=tracking_path, st_meta_path=meta_path).load_data()
print(soccertrack_df.head())
soccertrack_df.to_csv("117093_pp.csv", index=False)
