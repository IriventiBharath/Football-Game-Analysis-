from sklearn.cluster import KMeans
import numpy as np
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Clamp bounding box coordinates to ensure they are within frame bounds
        x_min = max(0, int(bbox[0]))
        y_min = max(0, int(bbox[1]))
        x_max = min(frame.shape[1], int(bbox[2]))
        y_max = min(frame.shape[0], int(bbox[3]))

        # Print bounding box information for debugging
        print(f"Original bbox: {bbox}")
        print(f"Clamped bbox: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        # Ensure the bounding box is valid
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounding box after clamping: {bbox}")

        # Crop the image based on the bounding box
        image = frame[y_min:y_max, x_min:x_max]

        # Check if the cropped image is black
        if np.all(image == 0):
            print("Warning: Cropped image is entirely black.")

        # Use the top half of the image to exclude the shorts/legs
        top_half_image = image[:image.shape[0] // 2, :]

        # Ensure the cropped region is not empty
        if top_half_image.size == 0:
            raise ValueError("Clamped bounding box resulted in an invalid or empty region.")

    # Display or save the cropped image for visual debugging
    # cv2.imshow('Cropped Image', top_half_image)  # Uncomment if using OpenCV
    # cv2.waitKey(0)

    # Proceed with color extraction and clustering
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
    
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color



    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id