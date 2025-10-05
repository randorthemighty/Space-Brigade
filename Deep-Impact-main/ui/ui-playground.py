from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTableWidgetItem,
    QCheckBox,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from deepimpack_UI import Ui_MainWindow
import deepimpact
import folium


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Import the UI class defined in the ui file
        self.ui = Ui_MainWindow()
        # Initialize the UI
        self.ui.setupUi(self)
        # table
        self.ui.table.setColumnWidth(0, 80)
        self.ui.table.setColumnWidth(1, 80)
        self.ui.table.setColumnWidth(2, 50)
        # checkbox
        self.ui.checkbox1 = QCheckBox(checked=True)
        self.ui.checkbox2 = QCheckBox(checked=True)
        self.ui.checkbox3 = QCheckBox(checked=True)
        self.ui.checkbox4 = QCheckBox(checked=True)
        self.ui.checkboxlist = [
            self.ui.checkbox1,
            self.ui.checkbox2,
            self.ui.checkbox3,
            self.ui.checkbox4,
        ]

        # check table

        item1 = QTableWidgetItem()
        item2 = QTableWidgetItem()
        item3 = QTableWidgetItem()
        item4 = QTableWidgetItem()
        self.ui.table.setItem(0, 2, item1)
        self.ui.table.setItem(1, 2, item2)
        self.ui.table.setItem(2, 2, item3)
        self.ui.table.setItem(3, 2, item4)
        self.ui.table.setCellWidget(0, 2, self.ui.checkbox1)
        self.ui.table.setCellWidget(1, 2, self.ui.checkbox2)
        self.ui.table.setCellWidget(2, 2, self.ui.checkbox3)
        self.ui.table.setCellWidget(3, 2, self.ui.checkbox4)

        # html interface

        self.ui.browser = QWebEngineView(self.ui.widget)
        self.ui.browser.setGeometry(
            self.ui.widget.x(),
            self.ui.widget.y(),
            self.ui.widget.width(),
            self.ui.widget.height(),
        )
        # layout.addWidget(self.ui.browser)
        print(self.ui.browser.size())
        print(self.ui.browser.size())
        # color list
        self.circle_list = ["green", "cornflowerblue", "pink", "red"]

        # plot
        self.plot = [True] * 4
        self.ui.checkbox1.stateChanged.connect(self.checkbox1)
        self.ui.checkbox2.stateChanged.connect(self.checkbox2)
        self.ui.checkbox3.stateChanged.connect(self.checkbox3)
        self.ui.checkbox4.stateChanged.connect(self.checkbox4)

        # button

        self.ui.generate_buttom.clicked.connect(self.button_generation)
        self.ui.plot_button.clicked.connect(self.plot_html)

    def setup_dropdown(self, entries):
        combo = QComboBox(self)
        combo.setEditable(True)
        completer = QCompleter(entries)
        completer.setCaseSensitivity(False)
        combo.setCompleter(completer)
        combo.addItems(entries)
        self.ui.dropdown_search = combo
        self.ui.layout.addWidget(combo)
        combo.currentTextChanged.connect(self.on_dropdown_selection)

    def on_dropdown_selection(self, text):
        # Use the selected text as input
        self.selected_dropdown_value = text

### this button generation code takes user input and compiles it in a list indexed as below. 
### later the list is used to call the deepimpact solver functions
    def button_generation(self: list) -> list:
        self.input = []
        # read data
        self.input = [
            float(self.ui.radius.text()),   ## input[0]
            float(self.ui.angle.text()),     ## input[1]
            float(self.ui.strength.text()),  ## input[2]
            float(self.ui.density.text()),   ## input[3]
            float(self.ui.velocity.text()),  ## input[4]
            float(self.ui.latitude.text()),  ## input[5]
            float(self.ui.longitude.text()), ## input[6]
            float(self.ui.bearing.text()),   ## input[7]
        ]


    

        # Generate result using deepimpact solver

### once we get the dropdown working, we would change the self statements to be the variables
### we get from NEoWs api. I believe we don't neet to use self in front of the variables, but
### i could be wrong.

        earth = deepimpact.Planet()
        result = earth.solve_atmospheric_entry(
            radius: float=self.input[0],
            angle: float=self.input[1],
            strength: float=self.input[2],
            density: float=self.input[3],
            velocity: float=self.input[4],
        )
        result: DataFrame = earth.calculate_energy(result: DataFrame)
        outcome: Dict = earth.analyse_outcome(result: DataFrame)

        # Calculate the blast location and damage radius for several pressure levels

### we will have to do some fancy work to get lat lon if its not provided in NEoWs api. might need
### to create a function to predict or calculate lat lon based on other data provided.
        pressures: List[float] = [1e3, 4e3, 30e3, 50e3]
        a = deepimpact.damage_zones(
            outcome: Dict,
            lat: float=self.input[5],
            lon: float=self.input[6],
            bearing: float=self.input[7],
            pressures: List[float]=pressures,
        ) -> blat: float, blon: float, damrad: List[float]:
### i believe the "a" list is a populated by the above solvers
        self.blast_lat = a[0]
        self.blast_lon = a[1]
        self.damage_rad = a[2]

        self.damage_rad_num: int = len(self.damage_rad)

        # Display type + zero pint + radius

        # zero point
        print(self.damage_rad)
        self.ui.type.clear()
        self.ui.zero_point1.clear()
        self.ui.zero_point2.clear()
        self.ui.type.append(outcome["outcome"])
        self.ui.zero_point1.append(str(self.blast_lat))
        self.ui.zero_point2.append(str(self.blast_lon))
        
        
        # radius
        #This iterates through every integer of the damage radius and populates the table for widget item.
        for ii in range(self.damage_rad_num):
            self.ui.table.setItem(ii, 1, QTableWidgetItem(str(self.damage_rad[ii])))
        
        # deal with checkbox with check or not
        for ii in range(4):
            if ii < self.damage_rad_num:
                self.ui.checkboxlist[ii].setChecked(True)
            else:
                self.ui.checkboxlist[ii].setChecked(False)

        self.input = []

    # Plots, based on the checkbox

    def plot_html(self):
        # map function takes parameters and moves map to that location.
        map: Map = folium.Map(
            location=[self.blast_lat, self.blast_lon], control_scale=True, zoom_start=7
        )
        # plots the damage circle on the maps
        for ii in range(self.damage_rad_num):
            if self.plot[ii]:
                folium.Circle(
                    [self.blast_lat, self.blast_lon],
                    self.damage_rad[ii],
                    color=self.circle_list[ii],
                    fill=True,
                    fillOpacity=0.1,
                ).add_to(map)

        # save
        map.save("ui_map.html")
        # read

        with open("./ui_map.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        self.ui.browser.setHtml(html_content)

    # checkbox change
    def checkbox1(self):
        self.plot[0] = not self.plot[0]

    def checkbox2(self):
        self.plot[1] = not self.plot[1]

    def checkbox3(self):
        self.plot[2] = not self.plot[2]

    def checkbox4(self):
        self.plot[3] = not self.plot[3]


app = QApplication([])
mainw = MainWindow()
mainw.show()
app.exec_()