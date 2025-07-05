import matplotlib.pyplot as plt

def show_single_image(image, label=None, prediction=None):
    
    plt.imshow(image.reshape(28, 28), cmap='gray')
    
    title = ""
    if label is not None:
        title += f"Label: {label}"
    if prediction is not None:
        title += f" | Predicted: {prediction}"

    plt.title(title.strip())
    plt.axis('off')
    plt.show()
