with open("web/templates/index.html", "r") as f:
    html = f.read()

old_str = """        let isGridView = true;

        function toggleView() {
            isGridView = !isGridView;
            const vid = document.getElementById('video-stream');
            const btn = document.getElementById('view-toggle-btn');
            
            vid.style.transition = 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            
            if (isGridView) {
                vid.style.transform = 'scale(1)';
                vid.style.transformOrigin = 'center center';
                btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l5-5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg> FOCUS FUSION`;
                btn.className = "px-8 py-2.5 w-full bg-white border-2 border-purple-600 text-purple-600 hover:bg-purple-600 hover:text-white font-bold rounded-md transition-all shadow-sm flex items-center justify-center gap-2";
            } else {
                vid.style.transform = 'scale(2)';
                vid.style.transformOrigin = '100% 100%'; // Focus on bottom right quadrant (AI Risk Fusion)
                btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg> SHOW GRID`;
                btn.className = "px-8 py-2.5 w-full bg-purple-600 text-white font-bold rounded-md transition-all shadow-md flex items-center justify-center gap-2";
            }
        }"""

new_str = """        let currentViewIdx = 4; // Starts on Grid View
        
        const presentationSequence = [
            { origin: "0% 0%", scale: 2, nextTarget: "HEATMAP", current: "YOLO" },          // 0: Focuses Top-Left (YOLO)
            { origin: "100% 0%", scale: 2, nextTarget: "OPTICAL FLOW", current: "HEATMAP" }, // 1: Focuses Top-Right (Heatmap)
            { origin: "0% 100%", scale: 2, nextTarget: "RISK FUSION", current: "FLOW" },    // 2: Focuses Bottom-Left (Flow)
            { origin: "100% 100%", scale: 2, nextTarget: "MULTI-GRID", current: "FUSION" }, // 3: Focuses Bottom-Right (Fusion)
            { origin: "center center", scale: 1, nextTarget: "YOLO DETECT", current: "GRID" }// 4: Shows Full Grid
        ];

        function toggleView() {
            currentViewIdx = (currentViewIdx + 1) % presentationSequence.length;
            const vid = document.getElementById('video-stream');
            const btn = document.getElementById('view-toggle-btn');
            const view = presentationSequence[currentViewIdx];
            
            vid.style.transition = 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            vid.style.transformOrigin = view.origin;
            vid.style.transform = `scale(${view.scale})`;
            
            let btnClass = "px-8 py-2.5 w-full border-2 font-bold rounded-md transition-all shadow-sm flex items-center justify-center gap-2 ";
            if (view.scale === 1) {
                btnClass += "bg-white border-purple-600 text-purple-600 hover:bg-purple-600 hover:text-white";
            } else {
                btnClass += "bg-purple-600 border-purple-600 text-white shadow-md";
            }
            btn.className = btnClass;
            
            btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg> FOCUS ${view.nextTarget}`;
        }"""

if old_str in html:
    html = html.replace(old_str, new_str)
    with open("web/templates/index.html", "w") as f:
        f.write(html)
    print("SUCCESS")
else:
    print("NOT FOUND")
