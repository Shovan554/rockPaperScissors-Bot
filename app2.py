import cv2

def main():
    # Load the response images
    paper_image = cv2.imread("paper.jpg")
    scissor_image = cv2.imread("scissor.jpeg")
    rock_image = cv2.imread("rock.jpg")

    # Verify that the images loaded successfully
    if paper_image is None:
        print("Error: paper.jpg not found!")
        return
    if scissor_image is None:
        print("Error: scissor.jpeg not found!")
        return
    if rock_image is None:
        print("Error: rock.jpg not found!")
        return

    # Create a window to display the images
    cv2.namedWindow("Response", cv2.WINDOW_NORMAL)

    # List of images and corresponding names
    images = [paper_image, scissor_image, rock_image]
    names = ["paper", "scissors", "rock"]

    index = 0
    while True:
        # Display the current image
        cv2.imshow("Response", images[index])
        print("Showing:", names[index])
        
        # Wait for 2000ms (2 seconds) or until 'q' is pressed
        key = cv2.waitKey(2000) & 0xFF
        if key == ord('q'):
            break
        
        # Cycle to the next image
        index = (index + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()