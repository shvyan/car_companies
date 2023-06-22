import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read national park data from the CSV file
Brands = []
with open('/content/vehicle_companies.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        brand = {
            'BRAND': row['BRAND'],
            'COMPANY': row['COMPANY'],
            'COSTS': row['COSTS']
        }
        Brands.append(brand)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in Brands:
    description = row['COSTS']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Serengeti National Park". We can recommend another national park based on cosine similarity.
liked_brand = "Toyota"
liked_brand_index = next(index for index, brand in enumerate(Brands) if brand['BRAND'] == liked_brand)

# Find the most similar national parks
similar_brand_indices = similarity_matrix[liked_brand_index].argsort()[::-1][1:3]  # Exclude the liked brand itself and get top 2 recommendations

recommended_brands = [Brands[index] for index in similar_brand_indices]

print("Because you liked " + liked_brand + ", we recommend the following brands:")

for recommended_brand in recommended_brands:
    print(recommended_brand['BRAND'])
