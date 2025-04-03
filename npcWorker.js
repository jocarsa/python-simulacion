// npcWorker.js
importScripts('utils.js', 'Persona.js');

let agent = null;
let terrainLayer = [];
let scaleFactor = 1;

onmessage = function(e) {
    const data = e.data;
    if (data.type === 'init') {
        // Initialize the Persona using data sent from the main thread
        agent = new Persona(
            data.agentData.id,
            data.agentData.initialPosition,
            data.agentData.bedPosition,
            data.agentData.workPosition,
            data.agentData.foodPosition
        );
        // If a saved state exists, restore its dynamic properties.
        // Exclude ephemeral properties (like target and path) to force recalculation.
        if (data.agentData.savedState) {
            Object.assign(agent, data.agentData.savedState);
            delete agent.target;
            agent.path = [];
        }
        terrainLayer = data.terrainLayer;
        scaleFactor = data.scaleFactor;
    } else if (data.type === 'update') {
        // Calculate current hour and day of the week from simulation time
        const currentTimeSeconds = data.currentTimeSeconds;
        const currentHour = Math.floor((currentTimeSeconds % 86400) / 3600);
        const dayOfWeek = [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
        ][Math.floor(currentTimeSeconds / 86400) % 7];

        // Update agent's need and move it based on the terrain layer and scale factor
        agent.updateNeed(currentHour, dayOfWeek);
        agent.move(terrainLayer, scaleFactor);

        // Send back the updated state to the main thread.
        // We include key properties so they can be saved to localStorage.
        postMessage({
            id: agent.id,
            position: agent.position,
            need: agent.need,
            bedPosition: agent.bedPosition,
            workPosition: agent.workPosition,
            foodPosition: agent.foodPosition,
            wcTimer: agent.wcTimer,
            path: agent.path
        });
    }
};

