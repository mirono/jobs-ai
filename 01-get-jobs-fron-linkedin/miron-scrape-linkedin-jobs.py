# Scrape LinkedIn jobs for the agent
# import urllib3
# from bs4 import BeautifulSoup
#
# def get_linkedin_jobs(pool: urllib3.PoolManager, start: int = 0) -> str:
#     jobs_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?&location=Israel&geoId=101620260&position=1&pageNum=0&start={start}"
#     response = pool.request("GET", jobs_url)
#     return response.data.decode()
#
# def get_linkedin_job(pool: urllib3.PoolManager, id: str) -> str:
#     job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{id}"
#     response = pool.request("GET", job_url)
#     return response.data.decode()
#
# http = urllib3.PoolManager()
# html = get_linkedin_jobs(http, 0)
# soup = BeautifulSoup(html,'html.parser')
# jobs = soup.find_all("li")
# for x in jobs:
#     jobid = x.find("div",{"class":"base-card"}).get('data-entity-urn').split(":")[3]
#     print(jobid)
#     html = get_linkedin_job(http, jobid)
#     with open(f"job-{jobid}.html", "w") as f:
#         f.write(html)

import logging
import os
import json

from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, \
    OnSiteOrRemoteFilters, SalaryBaseFilters

# Change root logger level (default is WARN)
logging.basicConfig(level=logging.INFO)


# Fired once for each successfully processed job
def on_data(data: EventData):
    print('[ON_DATA]', data.title, data.company, data.company_link, data.date, data.date_text, data.link, data.insights,
          len(data.description))
    with open(f"jobs/job-{data.job_id}.json", "w") as f:
        f.write(json.dumps(data._asdict()))


# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
    print('[ON_METRICS]', str(metrics))


def on_error(error):
    print('[ON_ERROR]', error)


def on_end():
    print('[ON_END]')


scraper = LinkedinScraper(
    chrome_executable_path=None,  # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
    chrome_binary_location=None,  # Custom path to Chrome/Chromium binary (e.g. /foo/bar/chrome-mac/Chromium.app/Contents/MacOS/Chromium)
    chrome_options=None,  # Custom Chrome options here
    headless=False,  # Overrides headless mode only if chrome_options is None
    max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
    # slow_mo=1.5,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
    slow_mo=20,
    page_load_timeout=40  # Page load timeout (in seconds)
)

# Add event listeners
scraper.on(Events.DATA, on_data)
scraper.on(Events.ERROR, on_error)
scraper.on(Events.END, on_end)

queries = [
    # Query(
    #     options=QueryOptions(
    #         limit=27  # Limit the number of jobs to scrape.
    #     )
    # ),
    Query(
        # query='Software',
        options=QueryOptions(
            locations=['Israel'],
            apply_link=True,  # Try to extract apply link (easy applies are skipped). If set to True, scraping is slower because an additional page must be navigated. Default to False.
            skip_promoted_jobs=False,  # Skip promoted jobs. Default to False.
            page_offset=2,  # How many pages to skip
            limit=12000,
            # filters=QueryFilters(
                # company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies.
                # relevance=RelevanceFilters.RECENT,
                # time=TimeFilters.ANY,
                # type=[TypeFilters.FULL_TIME],
                # on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE, OnSiteOrRemoteFilters.HYBRID, OnSiteOrRemoteFilters.ON_SITE],
                # experience=[ExperienceLevelFilters.MID_SENIOR],
                # base_salary=SalaryBaseFilters.SALARY_100K
            # )
        )
    ),
]

os.environ["LI_AT_COOKIE"] = "AQEDAQA....."
scraper.run(queries)