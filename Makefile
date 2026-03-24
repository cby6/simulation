.PHONY: install

install:
	bash scripts/init.sh

.PHONY: render

render:
	manim -p --resolution 854,480 --frame_rate 30 simulation/manim_traveling_magnetic_field.py ToroidalCoilFieldScene
