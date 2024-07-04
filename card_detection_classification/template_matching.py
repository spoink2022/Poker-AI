import cv2
import numpy as np

def detect_cards(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 200)

    # Contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for contour in contours:
        epsilon = 0.025 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            cards.append(approx)

    return cards

def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")

    s = rect.sum(axis=1)
    diff = np.diff(rect, axis=1)
    rect = np.array([rect[np.argmin(s)], rect[np.argmin(diff)], rect[np.argmax(s)], rect[np.argmax(diff)]])

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Load image
image = cv2.imread('card_detection_classification/image.jpg')
cv2.imshow('input image', image)

cards = detect_cards(image)

transformed_cards = [four_point_transform(image, card.reshape(4, 2)) for card in cards]

for i, card in enumerate(transformed_cards):
    cv2.imshow(f'Card {i+1}', card)
    cv2.waitKey(0)

def crop_rank(card):
    (h, w) = card.shape[:2]
    rank_region = card[0:int(h * 0.18), 5:int(w * 0.2)]  # Tuned
    return rank_region

def crop_suit(card):
    (h, w) = card.shape[:2]
    suit_region = card[int(h * 0.18):int(h * 0.3), 5:int(w * 0.2)]
    return suit_region

rank_regions = [crop_rank(card) for card in transformed_cards]
suit_regions = [crop_suit(card) for card in transformed_cards]

for i, (rank, suit) in enumerate(zip(rank_regions, suit_regions)):
    cv2.imshow(f'Rank {i+1}', rank)
    cv2.imshow(f'Suit {i+1}', suit)
    cv2.waitKey(0)

cv2.destroyAllWindows()

def template_matching(region, templates, threshold=0.5):
    best_match = None
    best_val = 0
    for template_name, template in templates.items():
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        cv2.imshow('test', region)
        cv2.imshow('test template', template)
        cv2.waitKey(0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > threshold and max_val > best_val:
            best_val = max_val
            best_match = template_name
    return best_match

rank_templates = {
    'A': cv2.imread('card_detection_classification/template_imgs/Ace.jpg', 0),
    'K': cv2.imread('card_detection_classification/template_imgs/King.jpg', 0),
    'Q': cv2.imread('card_detection_classification/template_imgs/Queen.jpg', 0),
    'J': cv2.imread('card_detection_classification/template_imgs/Jack.jpg', 0),
    '10': cv2.imread('card_detection_classification/template_imgs/Ten.jpg', 0),
    '9': cv2.imread('card_detection_classification/template_imgs/Nine.jpg', 0),
    '8': cv2.imread('card_detection_classification/template_imgs/Eight.jpg', 0),
    '7': cv2.imread('card_detection_classification/template_imgs/Seven.jpg', 0),
    '6': cv2.imread('card_detection_classification/template_imgs/Six.jpg', 0),
    '5': cv2.imread('card_detection_classification/template_imgs/Five.jpg', 0),
    '4': cv2.imread('card_detection_classification/template_imgs/Four.jpg', 0),
    '3': cv2.imread('card_detection_classification/template_imgs/Three.jpg', 0),
    '2': cv2.imread('card_detection_classification/template_imgs/Two.jpg', 0),
}

suit_templates = {
    'hearts': cv2.imread('card_detection_classification/template_imgs/Hearts.jpg', 0),
    'diamonds': cv2.imread('card_detection_classification/template_imgs/Diamonds.jpg', 0),
    'spades': cv2.imread('card_detection_classification/template_imgs/Spades.jpg', 0),
    'clubs': cv2.imread('card_detection_classification/template_imgs/Clubs.jpg', 0),
}

rank_regions_gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in rank_regions]
suit_regions_gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in suit_regions]

def extract_largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    return None

def crop_exact_region(region):
    _, binary = cv2.threshold(region, 128, 255, cv2.THRESH_BINARY_INV)
    
    rect = extract_largest_contour(binary)
    
    if rect:
        x, y, w, h = rect
        cropped_region = region[y:y+h, x:x+w]
        return cropped_region
    return region

rank_regions_exact = [crop_exact_region(region) for region in rank_regions_gray]
suit_regions_exact = [crop_exact_region(region) for region in suit_regions_gray]

def convert_to_binary(image):
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

rank_regions_binary = [convert_to_binary(region) for region in rank_regions_exact]
suit_regions_binary = [convert_to_binary(region) for region in suit_regions_exact]


def invert_image(image):
    return cv2.bitwise_not(image)

rank_regions_inverted = [invert_image(region) for region in rank_regions_exact]
suit_regions_inverted = [invert_image(region) for region in suit_regions_exact]

identified_ranks = [template_matching(region, rank_templates) for region in rank_regions_inverted]
identified_suits = [template_matching(region, suit_templates) for region in suit_regions_inverted]

identified_cards = list(zip(identified_ranks, identified_suits))

print(identified_cards)

cv2.destroyAllWindows()
