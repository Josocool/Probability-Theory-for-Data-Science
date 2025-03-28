import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 

# Data [Panin = Bream, Pakavor = Smelt]
Panin_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0] 
Panin_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 
                500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 
                680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0] 
Pakavor_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
Pakavor_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9] 

# Data Preparation
length = Panin_length + Pakavor_length 
weight = Panin_weight + Pakavor_weight  
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14  # Panin (Bream) = 1, Pakavor (Smelt) = 0

# Model Training
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target) 

def predict_fish():
    try:
        print("\n=== Fish Classification Program v2 ===")
        input_length = float(input('\nEnter fish length (cm): ')) 
        input_weight = float(input('Enter fish weight (g): ')) 
        
        # Prediction
        result = kn.predict([[input_length, input_weight]])
        
        # Output the results
        print('\nAnalysis Result:')
        if result[0] == 1:
            print('This fish is a Bream! (Confidence: {:.2f}%)'.format(
                kn.predict_proba([[input_length, input_weight]])[0][1] * 100))
        else: 
            print('This fish is a Smelt! (Confidence: {:.2f}%)'.format( 
                kn.predict_proba([[input_length, input_weight]])[0][0] * 100))
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(Panin_length, Panin_weight, c='blue', label='Bream', alpha=0.7)
        plt.scatter(Pakavor_length, Pakavor_weight, c='orange', label='Smelt', alpha=0.7)
        plt.scatter(input_length, input_weight, c='red', marker='*', s=300, label='Input')
        
        plt.xlabel('Length (cm)')
        plt.ylabel('Weight (g)') 
        plt.title('Fish Classification Result') 
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ValueError:
        print('\n Error: Please enter valid numbers.')
         
if __name__ == "__main__":
    predict_fish()

    
