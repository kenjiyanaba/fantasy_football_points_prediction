from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, csv

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

positions = ["qb", "rb", "wr", "te"]
years = list(range(2014, 2026))

all_players = []

for year in years:
    for position in positions:
        page = 1
        last_seen_players = set()
        print(f"🔍 Fetching {position.upper()} for {year}...")

        while True:
            url = f"https://www.nfl.com/draft/tracker/prospects/{position}/all-colleges/all-statuses/{year}?page={page}"
            driver.get(url)

            try:
                WebDriverWait(driver, 7).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
                )
                time.sleep(0.5)

                rows = driver.find_elements(By.CSS_SELECTOR, "tr")
                current_players = set()
                new_data = []

                for row in rows:
                    try:
                        player_link = row.find_element(By.CSS_SELECTOR, "a.css-1uiwkp4-Af")
                        name = player_link.text.strip()
                        grade = row.find_element(By.CSS_SELECTOR, "td.css-1a9pvo9-Af").text.strip()

                        current_players.add(name)

                        new_data.append({
                            "Year": year,
                            "Position": position.upper(),
                            "Player": name,
                            "Grade": grade
                        })
                    except Exception:
                        continue

                # Stop if current page is same as last (looping)
                if current_players == last_seen_players or len(current_players) == 0:
                    break

                last_seen_players = current_players
                all_players.extend(new_data)
                print(f"  ➕ Page {page}: {len(new_data)} players")

                page += 1

            except Exception as e:
                print(f"  ⚠️ Failed on page {page}: {e}")
                break

driver.quit()

with open("nfl_draft_prospects_full.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Year", "Position", "Player", "Grade"])
    writer.writeheader()
    writer.writerows(all_players)

print(f"\n✅ Scraping complete. Total players written: {len(all_players)}")
