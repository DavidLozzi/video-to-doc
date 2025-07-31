import httpx
import json
import re
from urllib.parse import urlparse

SERIES_URL = "https://freecomic.to/comic/star-wars-legacy-of-vader-2025.164413"

def extract_chapter_ids():
    """Extract chapter IDs from the series page"""
    print("Extracting chapter IDs from the series page...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'accept-language': 'en-US,en;q=0.9',
    }
    
    with httpx.Client(timeout=30.0) as client:
        response = client.get(SERIES_URL, headers=headers)
        response.raise_for_status()
        
        html_content = response.text
        
        # Look for chapter IDs in the HTML
        chapter_ids = []
        
        # Pattern 1: Look for [number] patterns
        id_matches = re.findall(r'\[(\d+)\]', html_content)
        for match in id_matches:
            if match not in chapter_ids:
                chapter_ids.append(match)
                print(f"Found chapter ID: {match}")
        
        # Pattern 2: Look for specific patterns that might be chapter IDs
        # Based on your curl example, the ID was 2779508
        # Let's look for similar patterns
        potential_ids = re.findall(r'(\d{7,})', html_content)  # 7+ digit numbers
        for potential_id in potential_ids:
            if potential_id not in chapter_ids:
                chapter_ids.append(potential_id)
                print(f"Found potential chapter ID: {potential_id}")
        
        return chapter_ids

def fetch_chapter_images(chapter_id):
    """Fetch images for a specific chapter using the curl pattern"""
    print(f"Fetching images for chapter ID: {chapter_id}")
    
    # Use the exact headers from your curl command
    headers = {
        'accept': 'text/x-component',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'no-cache',
        'content-type': 'text/plain;charset=UTF-8',
        'next-action': '7f5de9b5f0064b81b6fe299caf58f1f4b5897eed51',
        'next-router-state-tree': '%5B%22%22%2C%7B%22children%22%3A%5B%5B%22domain%22%2C%22freecomic.to%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22(routes)%22%2C%7B%22children%22%3A%5B%22series%22%2C%7B%22children%22%3A%5B%5B%22category%22%2C%22comic%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%5B%22slug%22%2C%22star-wars-legacy-of-vader-2025.164413%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2Fcomic%2Fstar-wars-legacy-of-vader-2025.164413%22%2C%22refresh%22%5D%7D%5D%7D%5D%7D%5D%7D%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D',
        'origin': 'https://freecomic.to',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://freecomic.to/comic/star-wars-legacy-of-vader-2025.164413',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
    }
    
    data = f'[{chapter_id}]'
    
    with httpx.Client(timeout=30.0) as client:
        try:
            response = client.post(SERIES_URL, headers=headers, data=data)
            response.raise_for_status()
            
            # Parse the response
            response_text = response.text
            print(f"Response preview: {response_text[:200]}...")
            
            # Parse the JSON response to extract image URLs
            try:
                # The response format is: "0:{...}\n1:[{...}]"
                response_lines = response_text.strip().split('\n')
                if len(response_lines) >= 2:
                    image_data_line = response_lines[1]
                    if image_data_line.startswith('1:'):
                        # Extract the JSON array part
                        json_part = image_data_line[2:]  # Remove "1:" prefix
                        image_array = json.loads(json_part)
                        
                        # Extract src URLs from each image object
                        image_urls = []
                        for image_obj in image_array:
                            if 'src' in image_obj:
                                image_urls.append(image_obj['src'])
                        
                        print(f"Found {len(image_urls)} images for chapter {chapter_id}")
                        for i, url in enumerate(image_urls[:5]):  # Show first 5
                            print(f"  Image {i+1}: {url}")
                        if len(image_urls) > 5:
                            print(f"  ... and {len(image_urls) - 5} more images")
                        
                        # Save chapter info
                        chapter_info = {
                            'id': chapter_id,
                            'title': f"Chapter {chapter_id}",
                            'image_urls': image_urls
                        }
                        
                        # Save to file
                        with open(f"chapter_{chapter_id}.json", "w") as f:
                            json.dump(chapter_info, f, indent=2)
                        print(f"Saved chapter info to chapter_{chapter_id}.json")
                        
                        return image_urls
                    else:
                        print("Unexpected response format")
                        
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                # Fallback to regex pattern matching
                imgor_pattern = r'https://imgor\.comick\.site/[^"\s]+'
                matches = re.findall(imgor_pattern, response_text)
                print(f"Found {len(matches)} imgor URLs using regex fallback")
                for i, url in enumerate(matches[:5]):
                    print(f"  Image {i+1}: {url}")
                return matches
                
        except Exception as e:
            print(f"Error fetching chapter {chapter_id}: {e}")
            return []

def main():
    print("Starting chapter extraction...")
    
    # Extract chapter IDs
    chapter_ids = extract_chapter_ids()
    
    if not chapter_ids:
        print("No chapter IDs found. Trying with known ID from your curl example...")
        # Use the ID from your curl example
        chapter_ids = ['2779508']
    
    print(f"\nFound {len(chapter_ids)} chapter IDs: {chapter_ids}")
    
    # Fetch images for each chapter
    print("\n" + "="*50)
    print("EXTRACTING IMAGES FOR EACH CHAPTER")
    print("="*50)
    
    all_chapters = []
    
    for chapter_id in chapter_ids:
        print(f"\nProcessing chapter ID: {chapter_id}")
        images = fetch_chapter_images(chapter_id)
        
        if images:
            chapter_data = {
                'id': chapter_id,
                'title': f"Chapter {chapter_id}",
                'image_urls': images
            }
            all_chapters.append(chapter_data)
    
    # Save all chapters to a single file
    if all_chapters:
        with open("all_chapters.json", "w") as f:
            json.dump(all_chapters, f, indent=2)
        print(f"\nSaved all {len(all_chapters)} chapters to all_chapters.json")
    
    print(f"\nExtraction complete! Found {len(all_chapters)} chapters with images.")

if __name__ == "__main__":
    main() 