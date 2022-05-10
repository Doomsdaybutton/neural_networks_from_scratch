gridX = 28;
gridY = 28;
gridSize = 10;
gridXOffset = 20;
gridYOffset = 20;
buttonOffset = 20;

let grid, mouse, resetButton, exportButton;

class Node {
	constructor(x, y, on = false) {
		this.x = x;
		this.y = y;
		this.on = on;
	}
}

class Mouse {
	constructor() {
		this.isAdding = false;
		this.prevX = this.x;
		this.prevY = this.y;
	}
	pressed(grid) {
		if (!this.onGrid) return;
		this.isAdding = !grid[this.x][this.y].on;
		grid[this.x][this.y].on = !grid[this.x][this.y].on;
		this.prevX = this.x;
		this.prevY = this.y;
	}
	dragged(grid) {
		if (this.onGrid && (this.prevX != this.X || this.prevY != this.Y)) {
			grid[this.x][this.y].on = this.isAdding;
		}
		this.prevX = this.x;
		this.prevY = this.y;
	}
	get x() {
		return Math.floor((mouseX - gridXOffset) / gridSize);
	}
	get y() {
		return Math.floor((mouseY - gridYOffset) / gridSize);
	}
	get onGrid() {
		return this.x > 0 && this.x < gridX && this.y > 0 && this.y < gridY;
	}
}

function resetGrid() {
	for (let x = 0; x < gridX; x++) {
		for (let y = 0; y < gridY; y++) {
			grid[x][y].on = false;
		}
	}
}

function exportGrid() {
	outGrid = grid.map((row) => {
		console.log(row);
		return row.map((node) => {
			if (node.on) return 1;
			return 0;
		});
	});

	let blob = new Blob([outGrid], { type: "text/csv" });
	var link = document.createElement("a");
	link.download = "test.txt";
	link.href = URL.createObjectURL(blob);
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
	delete link;
}

function drawGrid(grid) {
	stroke(0);
	fill(255);
	for (let x = 0; x < gridX; x++) {
		for (let y = 0; y < gridY; y++) {
			if (grid[x][y].on) fill(0);
			square(gridXOffset + x * gridSize, gridYOffset + y * gridSize, gridSize);
			fill(255);
		}
	}
}

function setup() {
	createCanvas(4000, 800);
	grid = new Array(gridX);
	for (let i = 0; i < gridX; i++) {
		grid[i] = new Array(gridY);
		for (let j = 0; j < gridY; j++) {
			grid[i][j] = new Node(i, j);
		}
	}

	mouse = new Mouse();

	resetButton = createButton("Reset");
	resetButton.position(
		gridXOffset + gridX * gridSize + buttonOffset,
		65 + gridYOffset
	);
	resetButton.mousePressed(resetGrid);

	exportButton = createButton("Export");
	exportButton.position(
		gridXOffset + gridX * gridSize + buttonOffset,
		130 + gridYOffset
	);
	exportButton.mousePressed(exportGrid);
}

function draw() {
	drawGrid(grid);
}

function mousePressed() {
	mouse.pressed(grid);
}

function mouseDragged() {
	mouse.dragged(grid);
}
