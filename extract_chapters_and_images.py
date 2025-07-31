import asyncio
import re
import json
import httpx
from pathlib import Path
from playwright.async_api import async_playwright

SERIES_URL = "https://freecomic.to/comic/star-wars-legacy-of-vader-2025.164413"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        chapter_ids = {}
        chapter_titles = {}

        # Intercept network to capture chapter POST payloads
        async def on_request(request):
            if request.method == "POST" and "star-wars-legacy-of-vader-2025.164413" in request.url:
                try:
                    post_data = request.post_data
                    if post_data:
                        # payload is like: [1234567]
                        m = re.match(r"\[(\d+)\]", post_data)
                        if m:
                            chapter_id = m.group(1)
                            print(f"Captured chapter ID payload: {post_data}")
                            chapter_ids[chapter_id] = post_data
                except Exception as e:
                    print(f"Error processing request: {e}")

        page.on("request", on_request)
        await page.goto(SERIES_URL, wait_until="networkidle")

        # Try to extract chapter information from the page state first
        print("Extracting chapter information from page state...")
        
        try:
            # Get the Next.js data
            next_data = await page.evaluate("window.__NEXT_DATA__")
            if next_data:
                print("Found __NEXT_DATA__, searching for chapter IDs...")
                # Look for chapter data in the Next.js state
                state_str = json.dumps(next_data)
                id_matches = re.findall(r'\[(\d+)\]', state_str)
                for match in id_matches:
                    if match not in chapter_ids:
                        chapter_ids[match] = f"[{match}]"
                        print(f"Found chapter ID in state: {match}")
        except Exception as e:
            print(f"Could not extract from __NEXT_DATA__: {e}")

        # If we still don't have chapter IDs, try to extract from the page content
        if not chapter_ids:
            print("Trying to extract from page content...")
            try:
                # Look for any script tags that might contain chapter data
                scripts = await page.query_selector_all("script")
                for script in scripts:
                    try:
                        content = await script.text_content()
                        if content and "star-wars-legacy-of-vader-2025" in content:
                            # Look for chapter IDs in the script
                            id_matches = re.findall(r'\[(\d+)\]', content)
                            for match in id_matches:
                                if match not in chapter_ids:
                                    chapter_ids[match] = f"[{match}]"
                                    print(f"Found chapter ID in script: {match}")
                    except:
                        continue
            except Exception as e:
                print(f"Error extracting from scripts: {e}")

        print(f"\nFound {len(chapter_ids)} chapter IDs: {list(chapter_ids.keys())}")

        if not chapter_ids:
            print("No chapters found. Exiting.")
            await browser.close()
            return

        # Now fetch images for each chapter
        print("\n" + "="*50)
        print("EXTRACTING IMAGES FOR EACH CHAPTER")
        print("="*50)

        for chapter_id in chapter_ids:
            print(f"\nProcessing chapter ID: {chapter_id}")
            
            try:
                # Get current headers from the page
                headers = await page.evaluate("""() => {
                    const nextData = window.__NEXT_DATA__;
                    return {
                        nextAction: nextData?.nextAction || null,
                        routerState: nextData?.tree ? JSON.stringify(nextData.tree) : null
                    };
                }""")
                
                print(f"Headers: {headers}")
                
                # Make the POST request to get chapter images
                component_response = await page.evaluate("""async (id, headers) => {
                    const res = await fetch(window.location.pathname, {
                        method: "POST",
                        headers: {
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": headers.nextAction || "",
                            "next-router-state-tree": headers.routerState || "",
                            "referer": window.location.href,
                            "origin": "https://freecomic.to"
                        },
                        body: `[${id}]`
                    });
                    return await res.text();
                }""", chapter_id, headers)
                
                # Parse the response to extract image URLs
                try:
                    # The response format is: "0:{...}\n1:[{...}]"
                    response_lines = component_response.strip().split('\n')
                    if len(response_lines) >= 2:
                        image_data_line = response_lines[1]
                        if image_data_line.startswith('1:'):
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
                                'title': chapter_titles.get(str(chapter_id), f"Chapter {chapter_id}"),
                                'image_urls': image_urls
                            }
                            
                            # Save to file
                            with open(f"chapter_{chapter_id}.json", "w") as f:
                                json.dump(chapter_info, f, indent=2)
                            print(f"Saved chapter info to chapter_{chapter_id}.json")
                            
                        else:
                            print("Unexpected response format")
                            
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response: {e}")
                    # Fallback to regex pattern matching
                    imgor_pattern = r'https://imgor\.comick\.site/[^"\s]+'
                    matches = re.findall(imgor_pattern, component_response)
                    print(f"Found {len(matches)} imgor URLs using regex fallback")
                    for i, url in enumerate(matches[:5]):
                        print(f"  Image {i+1}: {url}")
                        
            except Exception as e:
                print(f"Error processing chapter {chapter_id}: {e}")

        await browser.close()
        print(f"\nExtraction complete! Found {len(chapter_ids)} chapters.")

if __name__ == "__main__":
    asyncio.run(main()) 