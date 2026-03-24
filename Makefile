.PHONY: install

install:
	bash scripts/init.sh

.PHONY: render

render:
	manim -p --resolution 427,240 --frame_rate 30 --disable_caching simulation/manim_traveling_magnetic_field.py ToroidalCoilFieldScene
