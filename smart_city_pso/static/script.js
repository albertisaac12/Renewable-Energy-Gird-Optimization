// static/script.js
// Ultra-aggressive realtime dispatch client (ASCII only)

let livePolling = true;
let realtimePollTimer = null;
let latestRealtimeOpt = null; // {alloc: [s,w,h], fitness, ts}

const POLL_INTERVAL = 1000;
const REALTIME_POLL_INTERVAL = 300; // 300 ms aggressive
const MAX_RENEWABLE_CAPACITY = 100;

function log(msg) {
    const box = document.getElementById("logBox");
    const ts = new Date().toISOString().replace("T"," ").split(".")[0];
    box.textContent += "[" + ts + "] " + msg + "\n";
    box.scrollTop = box.scrollHeight;
}

// UI setters
function updateSourceLive(name, live) {
    document.getElementById(name + "Live").innerText = live.toFixed(2) + " MW";
}
function updateSourceApplied(name, applied, mode) {
    document.getElementById(name + "Applied").innerText = applied.toFixed(2) + " MW";
    document.getElementById(name + "Mode").innerText = "Mode: " + mode;
}

function gridStatus(demand, supply) {
    if (demand > MAX_RENEWABLE_CAPACITY) return "Underpowered (Expected)";
    const diff = Math.abs(supply - demand);
    const ratio = diff / demand;
    if (ratio <= 0.05) return "Stable";
    if (ratio <= 0.20) return "Warning";
    return "Critical";
}

// Poll live simulation state
function pollState() {
    if (!livePolling) return;
    fetch("/api/state")
        .then(r => r.json())
        .then(data => {
            const t = data.t_idx;
            const demand = Number(data.demand);
            const liveS = Number(data.solar_mean);
            const liveW = Number(data.wind_mean);
            const liveH = Number(data.hydro_mean);

            updateSourceLive("solar", liveS);
            updateSourceLive("wind", liveW);
            updateSourceLive("hydro", liveH);

            // Determine applied allocation
            const useOpt = document.getElementById("applyOptimized").checked;
            let appliedS = liveS, appliedW = liveW, appliedH = liveH;
            let mode = "Live";

            if (useOpt && latestRealtimeOpt && latestRealtimeOpt.alloc) {
                // Use latest realtime optimized allocation
                appliedS = Number(latestRealtimeOpt.alloc[0]);
                appliedW = Number(latestRealtimeOpt.alloc[1]);
                appliedH = Number(latestRealtimeOpt.alloc[2]);
                mode = "Realtime-Optimized";
                document.getElementById("realtimeFitness").innerText = "Realtime fitness: " + Number(latestRealtimeOpt.fitness).toFixed(3);
            } else {
                document.getElementById("realtimeFitness").innerText = "Realtime fitness: -";
            }

            const total = appliedS + appliedW + appliedH;

            document.getElementById("timeIdx").innerText = "Time: " + t;
            document.getElementById("demand").innerText = "Demand: " + demand.toFixed(2) + " MW";
            document.getElementById("supply").innerText = "Applied Supply: " + total.toFixed(2) + " MW";

            updateSourceApplied("solar", appliedS, mode);
            updateSourceApplied("wind", appliedW, mode);
            updateSourceApplied("hydro", appliedH, mode);

            const st = gridStatus(demand, total);
            document.getElementById("gridStatus").innerText = "Grid Status: " + st;

            log("t=" + t + " Demand=" + demand.toFixed(2) + " Applied=" + total.toFixed(2) + " Status=" + st);
        })
        .finally(() => {
            if (livePolling) setTimeout(pollState, POLL_INTERVAL);
        });
}

// Realtime optimize call
function callRealtimeOptimize() {
    fetch("/api/optimize_now", { method: "POST" })
        .then(r => r.json())
        .then(json => {
            if (json && json.alloc) {
                latestRealtimeOpt = {
                    alloc: json.alloc,
                    fitness: json.fitness,
                    ts: Date.now() / 1000
                };
                // Do not flood logs - log only occasional messages
                log("Realtime alloc received, fitness: " + Number(json.fitness).toFixed(3));
            }
        })
        .catch(err => {
            log("Realtime optimize error: " + String(err));
        });
}

function startRealtimePolling() {
    if (realtimePollTimer) return;
    callRealtimeOptimize();
    realtimePollTimer = setInterval(callRealtimeOptimize, REALTIME_POLL_INTERVAL);
}

function stopRealtimePolling() {
    if (!realtimePollTimer) return;
    clearInterval(realtimePollTimer);
    realtimePollTimer = null;
}

// Checkbox toggle
document.getElementById("applyOptimized").onchange = function () {
    const checked = document.getElementById("applyOptimized").checked;
    if (checked) {
        // enable realtime optimize polling
        startRealtimePolling();
        log("Realtime optimized dispatch enabled");
    } else {
        stopRealtimePolling();
        latestRealtimeOpt = null;
        document.getElementById("realtimeFitness").innerText = "Realtime fitness: -";
        log("Realtime optimized dispatch disabled");
    }
};

// Pause/resume live
document.getElementById("pauseLive").onclick = function () {
    livePolling = false;
    stopRealtimePolling();
    log("Live polling paused");
};
document.getElementById("resumeLive").onclick = function () {
    if (!livePolling) {
        livePolling = true;
        pollState();
        log("Live polling resumed");
    }
};

// Restart
document.getElementById("restartSim").onclick = function () {
    fetch("/api/restart", { method: "POST" })
        .then(r => r.json())
        .then(json => {
            log("Simulation restarted");
            latestRealtimeOpt = null;
            stopRealtimePolling();
            document.getElementById("applyOptimized").checked = false;
            document.getElementById("realtimeFitness").innerText = "Realtime fitness: -";
            document.getElementById("logBox").textContent = "";
            updateSourceApplied("solar", 0, "Live");
            updateSourceApplied("wind", 0, "Live");
            updateSourceApplied("hydro", 0, "Live");
            livePolling = true;
            pollState();
        });
};

// Enable checkbox after initial UI ready and start polling
window.onload = function () {
    // enable optimized checkbox for user
    document.getElementById("applyOptimized").disabled = false;
    pollState();
    log("UI ready - live polling started");
};
