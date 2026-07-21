# 3D CNN + PointNet for Groundwater Arsenic Risk Prediction

This project develops a deep learning framework for predicting groundwater arsenic contamination risk using a combination of **3D Convolutional Neural Networks (3D CNNs)** and **PointNet-based spatial learning**.

The objective is to predict arsenic concentration and classify groundwater wells into different contamination risk categories by learning from both **local groundwater measurements** and **environmental/geological factors**.

## Project Overview

Groundwater arsenic contamination is influenced by complex interactions between geology, hydrology, sediment characteristics, groundwater depth, and local environmental conditions. Traditional models often rely on tabular features from individual wells, limiting their ability to capture spatial relationships.

This project introduces a multimodal deep learning approach:

- **PointNet branch:** Learns relationships between neighbouring groundwater wells, including spatial distance, depth variation, and local arsenic patterns.
- **3D CNN branch:** Learns environmental context from raster-based geological and geographical data such as elevation, lithology, land use, hydrology, and other environmental factors.
- **Fusion network:** Combines both representations to create a unified prediction model capable of using both local observations and broader environmental information.

The final aim is to improve groundwater arsenic risk mapping by combining observed well data with spatial environmental information.
The PointNet model performs well at learning local groundwater patterns, particularly for identifying high-risk wells, but struggles with ambiguous medium-risk regions where environmental context becomes important.
