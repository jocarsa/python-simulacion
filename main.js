// main.js

// Canvas and context setup
const backgroundCanvas = document.getElementById('backgroundCanvas');
const backgroundCtx = backgroundCanvas.getContext('2d');
const playersCanvas = document.getElementById('playersCanvas');
const playersCtx = playersCanvas.getContext('2d');
const statsDiv = document.getElementById('stats');
const clockDiv = document.getElementById('clock');
const pieChartCanvas = document.getElementById('pieChart');
const pieChartCtx = pieChartCanvas.getContext('2d');

const scaleFactor = 1;
const numAgents = 100;
let currentTimeSeconds = 0;
const timeStepSeconds = 20;
let terrainLayer = [];

// Global variables for state management and NPC workers
let savedStateFromServer = null;
let npcWorkers = [];
let npcStates = {};

// Create an Image object for the map
const mapa = new Image();
mapa.src = "casas.png";

// ----------------------
// Server State Functions
// ----------------------

// Load state from the server (GET request)
function loadStateFromServer() {
    return fetch('npc_endpoint.php')
    .then(response => response.json())
    .then(data => {
        if (data.npcStates && data.currentTimeSeconds !== undefined) {
            savedStateFromServer = data;
            npcStates = data.npcStates;
            currentTimeSeconds = data.currentTimeSeconds;
            console.log('Loaded state from server:', data);
        } else {
            console.log('No saved state on server; starting fresh.');
        }
    })
    .catch(error => {
        console.error('Error loading state:', error);
    });
}

// Save state to the server (POST request)
function saveStateToServer() {
    const stateToSave = {
        currentTimeSeconds: currentTimeSeconds,
        npcStates: npcStates
    };
    fetch('npc_endpoint.php', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(stateToSave)
    })
    .then(response => response.json())
    .then(function(data){
    //console.log('Saved state:', data)
    })
    .catch(error => console.error('Error saving state:', error));
}

// ----------------------
// Map and Terrain Setup
// ----------------------

// Once the map image loads, set up canvas dimensions and build the terrain matrix.
mapa.onload = function() {
    backgroundCanvas.width = mapa.width;
    backgroundCanvas.height = mapa.height;
    playersCanvas.width = mapa.width;
    playersCanvas.height = mapa.height;
    
    // Draw the map image on the background canvas.
    backgroundCtx.drawImage(mapa, 0, 0);
    
    // Build the terrainLayer matrix from the image data.
    const imageData = backgroundCtx.getImageData(0, 0, mapa.width, mapa.height).data;
    terrainLayer = [];
    for (let y = 0; y < mapa.height; y++) {
        const row = [];
        for (let x = 0; x < mapa.width; x++) {
            const index = (y * mapa.width + x) * 4;
            const r = imageData[index];
            const g = imageData[index + 1];
            const b = imageData[index + 2];
            let color = 'unknown';
            if (r === 0 && g === 255 && b === 0) color = 'green';
            if (r === 255 && g === 0 && b === 255) color = 'magenta';
            if (r === 0 && g === 255 && b === 255) color = 'yellow';
            if (r === 255 && g === 0 && b === 0) color = 'blue';
            if (r === 0 && g === 0 && b === 255) color = 'red';
            if (r === 255 && g === 255 && b === 0) color = 'cyan';
            if (r === 127 && g === 127 && b === 127) color = 'gray';
            if (r === 0 && g === 200 && b === 0) color = 'dark_green';
            row.push(color);
        }
        terrainLayer.push(row);
    }
    
    // Load any saved state from the server, then initialize agents and start the game loop.
    loadStateFromServer().then(() => {
        initializeAgents();
        gameLoop();
    });
};

// ----------------------
// Agent Initialization
// ----------------------

function initializeAgents() {
    // Use saved state from the server if available.
    const savedStates = (savedStateFromServer && savedStateFromServer.npcStates) ? savedStateFromServer.npcStates : null;
    // currentTimeSeconds has already been set if a saved state was loaded.
    
    // Build arrays for different terrain positions.
    const walkablePositions = [];
    const bedPositions = [];
    const workPositions = [];
    const foodPositions = [];

    for (let y = 0; y < terrainLayer.length; y++) {
        for (let x = 0; x < terrainLayer[y].length; x++) {
            const color = terrainLayer[y][x];
            if (color === 'green') walkablePositions.push([y, x]);
            if (color === 'blue') bedPositions.push([y, x]);
            if (color === 'gray') workPositions.push([y, x]);
            if (color === 'yellow') foodPositions.push([y, x]);
        }
    }

    npcWorkers = [];
    // If there is a saved state, use it; otherwise start with an empty object.
    npcStates = savedStates ? savedStates : {};

    for (let i = 0; i < numAgents; i++) {
        let initData;
        if (savedStates && savedStates[i]) {
            const savedAgent = savedStates[i];
            initData = {
                id: i,
                initialPosition: savedAgent.position,
                bedPosition: savedAgent.bedPosition || bedPositions[Math.floor(Math.random() * bedPositions.length)],
                workPosition: savedAgent.workPosition || workPositions[Math.floor(Math.random() * workPositions.length)],
                foodPosition: savedAgent.foodPosition || foodPositions[Math.floor(Math.random() * foodPositions.length)],
                savedState: savedAgent
            };
        } else {
            initData = {
                id: i,
                initialPosition: walkablePositions[Math.floor(Math.random() * walkablePositions.length)],
                bedPosition: bedPositions[Math.floor(Math.random() * bedPositions.length)],
                workPosition: workPositions[Math.floor(Math.random() * workPositions.length)],
                foodPosition: foodPositions[Math.floor(Math.random() * foodPositions.length)]
            };
        }

        const worker = new Worker('npcWorker.js');
        worker.postMessage({
            type: 'init',
            agentData: initData,
            terrainLayer: terrainLayer,
            scaleFactor: scaleFactor
        });

        // Update the global npcStates object when a worker sends a message.
        worker.onmessage = function(e) {
            const data = e.data;
            npcStates[data.id] = data;
        };

        npcWorkers.push(worker);
    }
}

// ----------------------
// Update, Draw and Loop
// ----------------------

// Update all NPCs by sending them an update message.
function update() {
    npcWorkers.forEach(worker => {
        worker.postMessage({ type: 'update', currentTimeSeconds: currentTimeSeconds });
    });
    currentTimeSeconds = (currentTimeSeconds + timeStepSeconds) % (12 * 31 * 86400);
}

// Draw each NPC's current position on the players canvas.
function draw() {
    Object.values(npcStates).forEach(state => {
        playersCtx.fillStyle = 'black';
        playersCtx.fillRect(state.position[1] * scaleFactor, state.position[0] * scaleFactor, scaleFactor, scaleFactor);
    });
}

// Clear the NPC canvas once per second to reduce trail buildup.
setInterval(() => {
    playersCtx.fillStyle = "rgba(255,255,255,0.1)";
    playersCtx.fillRect(0, 0, playersCanvas.width, playersCanvas.height);
    //console.log(Object.values(npcStates));
}, 1000);

// Update statistics display.
function updateStats() {
    let statsHTML = '';
    Object.values(npcStates).forEach(state => {
        statsHTML += `Agente ${state.id}: Pos (${state.position[1]},${state.position[0]}), Necesidad: ${state.need}<br>`;
    });
    statsDiv.innerHTML = statsHTML;
}

// Update clock display using the secondsToDateTimeStr function from utils.js.
function updateClock() {
    const dateTimeStr = secondsToDateTimeStr(currentTimeSeconds);
    clockDiv.innerHTML = `Fecha y Hora: ${dateTimeStr}`;
}

// Update pie chart displaying the distribution of NPC needs.
function updatePieChart() {
    const needsCounts = {
        moving: 0,
        food: 0,
        rest: 0,
        wc: 0,
        resting: 0,
        work: 0
    };

    Object.values(npcStates).forEach(state => {
        if (state.path && state.path.length > 0) {
            needsCounts.moving++;
        } else {
            needsCounts[state.need] = (needsCounts[state.need] || 0) + 1;
        }
    });

    const labels = Object.keys(needsCounts);
    const data = Object.values(needsCounts);
    const colors = ['orange', 'yellow', 'blue', 'red', 'cyan', 'gray'];

    pieChartCtx.clearRect(0, 0, pieChartCanvas.width, pieChartCanvas.height);
    pieChartCtx.fillStyle = 'white';
    pieChartCtx.fillRect(0, 0, pieChartCanvas.width, pieChartCanvas.height);

    let startAngle = 0;
    labels.forEach((label, index) => {
        const sliceAngle = (data[index] / numAgents) * 2 * Math.PI;
        pieChartCtx.beginPath();
        pieChartCtx.moveTo(200, 200);
        pieChartCtx.arc(200, 200, 200, startAngle, startAngle + sliceAngle);
        pieChartCtx.closePath();
        pieChartCtx.fillStyle = colors[index];
        pieChartCtx.fill();
        startAngle += sliceAngle;
    });
}

// Main game loop.
function gameLoop() {
    update();
    draw();
    updateStats();
    updateClock();
    updatePieChart();
    requestAnimationFrame(gameLoop);
}

// ----------------------
// Periodically Save State
// ----------------------

// Save the simulation state to the server every second.
setInterval(saveStateToServer, 1000);

