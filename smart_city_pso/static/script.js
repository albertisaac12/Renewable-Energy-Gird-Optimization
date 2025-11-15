// Real-time UI client for PSO-optimized dispatch

let livePolling = true;
let realtimePollTimer = null;
let latestRealtimeOpt = null;

const POLL_INTERVAL = 1000;
const REALTIME_INTERVAL = 300;

function log(msg) {
    const box = document.getElementById("logBox");
    const ts = new Date().toISOString().replace("T", " ").split(".")[0];
    box.textContent += "[" + ts + "] " + msg + "\n";
    box.scrollTop = box.scrollHeight;
}

function updateSourceLive(name, val) {
    document.getElementById(name + "Live").innerText = val.toFixed(2) + " MW";
}

function updateSourceApplied(name, val, mode) {
    document.getElementById(name + "Applied").innerText = val.toFixed(2) + " MW";
    document.getElementById(name + "Mode").innerText = "Mode: " + mode;
}

function gridStatus(demand, supply) {
    const diff = Math.abs(supply - demand);
    const ratio = diff / demand;
    if (ratio <= 0.05) return "Stable";
    if (ratio <= 0.20) return "Warning";
    return "Critical";
}

function updatePhase(t) {
    const el = document.getElementById("simPhase");

    if (t < 30) {
        el.innerText = "Simulation Phase: Stabilization Period (Demand ~95 MW)";
        el.className = "phaseBadge phaseBlue";
    } else {
        el.innerText = "Simulation Phase: Dynamic Demand (PSO Following Fluctuations)";
        el.className = "phaseBadge phaseGreen";
    }
}

function pollState() {
    if (!livePolling) return;

    fetch("/api/state")
        .then(r => r.json())
        .then(data => {
            const t = data.t_idx;
            const demand = Number(data.demand);

            const sLive = Number(data.solar_mean);
            const wLive = Number(data.wind_mean);
            const hLive = Number(data.hydro_mean);

            updateSourceLive("solar", sLive);
            updateSourceLive("wind", wLive);
            updateSourceLive("hydro", hLive);

            updatePhase(t);
            document.getElementById("timeIdx").innerText = "Time: " + t;
            document.getElementById("demand").innerText = "Demand: " + demand.toFixed(2) + " MW";

            const optOn = document.getElementById("applyOptimized").checked;
            let aS = sLive, aW = wLive, aH = hLive;
            let mode = "Live";

            if (optOn && latestRealtimeOpt && latestRealtimeOpt.alloc) {
                aS = Number(latestRealtimeOpt.alloc[0]);
                aW = Number(latestRealtimeOpt.alloc[1]);
                aH = Number(latestRealtimeOpt.alloc[2]);
                mode = "Realtime-Optimized";

                document.getElementById("realtimeFitness").innerText =
                    "Realtime fitness: " + latestRealtimeOpt.fitness.toFixed(3);
            } else {
                document.getElementById("realtimeFitness").innerText = "Realtime fitness: -";
            }

            const total = aS + aW + aH;

            updateSourceApplied("solar", aS, mode);
            updateSourceApplied("wind", aW, mode);
            updateSourceApplied("hydro", aH, mode);

            document.getElementById("supply").innerText = "Applied Supply: " + total.toFixed(2) + " MW";
            document.getElementById("gridStatus").innerText = "Grid Status: " + gridStatus(demand, total);

            log("t=" + t + " Demand=" + demand.toFixed(2) + " Supply=" + total.toFixed(2));
        })
        .finally(() => {
            if (livePolling) setTimeout(pollState, POLL_INTERVAL);
        });
}

function callRealtimeOptimize() {
    fetch("/api/optimize_now", { method: "POST" })
        .then(r => r.json())
        .then(json => {
            if (!json.alloc) return;
            latestRealtimeOpt = json;
        });
}

function startRealtimePolling() {
    if (!realtimePollTimer) {
        callRealtimeOptimize();
        realtimePollTimer = setInterval(callRealtimeOptimize, REALTIME_INTERVAL);
    }
}

function stopRealtimePolling() {
    if (realtimePollTimer) {
        clearInterval(realtimePollTimer);
        realtimePollTimer = null;
    }
}

document.getElementById("applyOptimized").onchange = function () {
    if (this.checked) startRealtimePolling();
    else {
        stopRealtimePolling();
        latestRealtimeOpt = null;
    }
};

document.getElementById("pauseLive").onclick = function () {
    livePolling = false;
    log("Live polling paused");
};

document.getElementById("resumeLive").onclick = function () {
    if (!livePolling) {
        livePolling = true;
        pollState();
        log("Live polling resumed");
    }
};

document.getElementById("restartSim").onclick = function () {
    fetch("/api/restart", { method: "POST" })
        .then(() => {
            latestRealtimeOpt = null;
            stopRealtimePolling();
            document.getElementById("applyOptimized").checked = false;
            document.getElementById("logBox").textContent = "";
            document.getElementById("realtimeFitness").innerText = "Realtime fitness: -";
            pollState();
            log("Simulation restarted");
        });
};

window.onload = function () {
    document.getElementById("applyOptimized").disabled = false;
    pollState();
    log("UI loaded - starting simulation");
};
