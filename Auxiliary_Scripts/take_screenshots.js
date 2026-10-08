const puppeteer = require('puppeteer');
const fs = require('fs');

async function run() {
    console.log("Launching browser...");
    const browser = await puppeteer.launch({ headless: true, defaultViewport: { width: 1440, height: 900 }});
    const page = await browser.newPage();
    
    console.log("Navigating to dashboard...");
    await page.goto('http://127.0.0.1:5001');
    
    // Screenshot 1: Idle
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: '/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/graphs/Final_Dash_1.png' });
    console.log("Saved Final_Dash_1.png (Idle)");

    // Load video test.mp4
    console.log("Loading Sparse Video test.mp4...");
    await page.evaluate(() => {
        fetch('/api/set_source', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: 'videos/test.mp4' }) });
    });
    
    // Screenshot 2: Sparse (YOLO active)
    await new Promise(r => setTimeout(r, 4000));
    await page.screenshot({ path: '/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/graphs/Final_Dash_2.png' });
    console.log("Saved Final_Dash_2.png (Sparse)");

    // Load video df.mp4 (Dense/Anomaly)
    console.log("Loading Dense Video df.mp4...");
    await page.evaluate(() => {
        fetch('/api/set_source', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: 'videos/df.mp4' }) });
    });
    
    // Screenshot 3: Dense CSRNet 
    await new Promise(r => setTimeout(r, 5000));
    await page.screenshot({ path: '/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/graphs/Final_Dash_3.png' });
    console.log("Saved Final_Dash_3.png (Dense)");

    // Wait until ~18-20s for Anomaly spike
    console.log("Waiting for High Risk threshold...");
    await new Promise(r => setTimeout(r, 23000)); 
    await page.screenshot({ path: '/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/graphs/Final_Dash_4.jpg', type: 'jpeg', quality: 95 });
    console.log("Saved Final_Dash_4.jpg (Anomaly)");

    await browser.close();
    console.log("Successfully captured all thesis dashboard screenshots.");
}

run();
