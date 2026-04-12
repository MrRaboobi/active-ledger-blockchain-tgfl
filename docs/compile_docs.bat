@echo off
echo Compiling Phase 3.1 Experimentation Report...
echo.
echo === Part 1 ===
pdflatex -interaction=nonstopmode phase3_experimentation_part1.tex
pdflatex -interaction=nonstopmode phase3_experimentation_part1.tex
echo.
echo === Part 2 ===
pdflatex -interaction=nonstopmode phase3_experimentation_part2.tex
pdflatex -interaction=nonstopmode phase3_experimentation_part2.tex
echo.
echo Done! Check phase3_experimentation_part1.pdf and phase3_experimentation_part2.pdf
pause
