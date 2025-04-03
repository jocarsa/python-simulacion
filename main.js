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

const mapa = new Image();
mapa.src = "casas.png";

mapa.onload = function() {
    backgroundCanvas.width = mapa.width;
    backgroundCanvas.height = mapa.height;
    playersCanvas.width = mapa.width;
    playersCanvas.height = mapa.height;
    backgroundCtx.drawImage(mapa, 0, 0);

    // Obtener la matriz de colores
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

    initializeAgents();
    gameLoop();
};

let agents = [];

function initializeAgents() {
    const walkablePositions = [];
    const bedPositions = [];
    const workPositions = [];
    const foodPositions = [];
    const parkPositions = [];

    for (let y = 0; y < terrainLayer.length; y++) {
        for (let x = 0; x < terrainLayer[y].length; x++) {
            const color = terrainLayer[y][x];
            if (color === 'green') walkablePositions.push([y, x]);
            if (color === 'blue') bedPositions.push([y, x]);
            if (color === 'gray') workPositions.push([y, x]);
            if (color === 'yellow') foodPositions.push([y, x]);
            if (color === 'dark_green') parkPositions.push([y, x]);
        }
    }

    agents = [];
    for (let i = 0; i < numAgents; i++) {
        const initialPosition = walkablePositions[Math.floor(Math.random() * walkablePositions.length)];
        const bedPosition = bedPositions[Math.floor(Math.random() * bedPositions.length)];
        const workPosition = workPositions[Math.floor(Math.random() * workPositions.length)];
        const foodPosition = foodPositions[Math.floor(Math.random() * foodPositions.length)];
        agents.push(new Persona(i, initialPosition, bedPosition, workPosition, foodPosition));
    }
}

function draw() {
    playersCtx.clearRect(0, 0, playersCanvas.width, playersCanvas.height);

    agents.forEach(agent => {
        playersCtx.fillStyle = 'black';
        playersCtx.fillRect(agent.position[1] * scaleFactor, agent.position[0] * scaleFactor, scaleFactor, scaleFactor);
    });
}

function update() {
    const currentHour = Math.floor((currentTimeSeconds % 86400) / 3600);
    const dayOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][Math.floor(currentTimeSeconds / 86400) % 7];

    agents.forEach(agent => {
        agent.updateNeed(currentHour, dayOfWeek);
        agent.move(terrainLayer, scaleFactor);
    });

    currentTimeSeconds = (currentTimeSeconds + timeStepSeconds) % (12 * 31 * 86400);
}

function updateStats() {
    statsDiv.innerHTML = agents.map(agent => `Agente ${agent.id}: Pos (${agent.position[1]},${agent.position[0]}), Necesidad: ${agent.need}`).join('<br>');
}

function updateClock() {
    const dateTimeStr = secondsToDateTimeStr(currentTimeSeconds);
    clockDiv.innerHTML = `Fecha y Hora: ${dateTimeStr}`;
}

function updatePieChart() {
    const needsCounts = {
        moving: 0,
        food: 0,
        rest: 0,
        wc: 0,
        resting: 0,
        work: 0
    };

    agents.forEach(agent => {
        if (agent.path.length > 0) {
            needsCounts.moving++;
        } else {
            needsCounts[agent.need]++;
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
        const sliceAngle = (data[index] / agents.length) * 2 * Math.PI;
        pieChartCtx.beginPath();
        pieChartCtx.moveTo(200, 200);
        pieChartCtx.arc(200, 200, 200, startAngle, startAngle + sliceAngle);
        pieChartCtx.closePath();
        pieChartCtx.fillStyle = colors[index];
        pieChartCtx.fill();
        startAngle += sliceAngle;
    });
}

function gameLoop() {
    update();
    draw();
    updateStats();
    updateClock();
    updatePieChart();
    requestAnimationFrame(gameLoop);
}

