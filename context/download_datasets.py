import undetected_chromedriver as uc
driver = uc.Chrome(
    headless=False,
    use_subprocess=False,
    browser_executable_path="/System/Volumes/Data/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
driver.get('https://app.roboflow.com/')

input("Login and press ENTER")

driver.save_screenshot('roboflow.png')