from requests_html import HTMLSession
import sys

script = """
    () => {
        const links = document.querySelectorAll('a[href^="gen9randombattle-"][class="blocklink"]');
        const hrefs = [];
        links.forEach(link => {
            if (/^gen9randombattle-\\d+$/.test(link.getAttribute('href'))) {
                hrefs.push(link.href);
            }
        });
        return hrefs;
    }
"""
session = HTMLSession()

def get_battles(page, retry=5):
    print(f"Page {page}: attempting to scrape")
    for _ in range(retry):
        url = f"https://replay.pokemonshowdown.com/?format=gen9randombattle&page={page}&sort=rating"
        resp = session.get(url)
        # https://requests-html.kennethreitz.org/ 
        # Docs to play around with parameters
        result = resp.html.render(script=script, wait=1, sleep=2) 
        if len(result) > 0:
            print(f"Page {page}: successfully scraped!")
            return result
        else:
            print(f"Page {page}: No results, retrying...")
    print(f"Page {page}: Unable to get results for page {page}")
    return []

all_results = []
if len(sys.argv) == 1:
    print("Checking pages 1 to 100...")
    pages = range(1, 101)
elif len(sys.argv) == 2:
    print(f"Checking pages 1 to {sys.argv[1]}...")
    pages = range(1, int(sys.argv[1])+1)
elif len(sys.argv) == 3:
    print(f"Checking pages {sys.argv[1]} to {sys.argv[2]}...")
    pages = range(int(sys.argv[1]), int(sys.argv[2])+1)

for page in pages:
    all_results.extend(get_battles(page))

with open('data/tmp/log_urls.txt', 'w+') as f:
    for result in all_results:
        f.write(result+'.log\n')
