# Probabilities Estimator for Dice Rolls in the Vampire the Masquerade 5th Edition (VtM V5) Tabletop Roleplaying Game

This is a personal project, born of my frustration in trying (and failing) to ascertain the probability distribution linked to a given dice roll in VtM v5, due to all the different variables (Hunger, Criticals, Messy Criticals, Bestial Failures, Willpower Rerolls...).  
It is also an excuse for me to experiment with strong typing and properly documented code in Python.

It relies on the [Law of Large Numbers (LLN)](https://en.wikipedia.org/wiki/Law_of_large_numbers), which basically says that if a trial has an associated (fixed) probability, the average of all trial results will tends towards the expected value of the trial.  
In the case of a TTRPG dice roll with a fixed set of outcomes (i.e. Success, Failure, Critical Success, Messy Critical \& Bestial Failure in VtM v5), it means that given a large enough number of simulated tests, the proportion of rolls with a given outcome would be an accurate estimation of the outcome's associated probability.  
I.e. that be how statistics are. If a thing has a probability, proportions derived from observations can determine it.  
*(This isn't meant as a complete and rigorous explanation... If any probability theorists or statisticians are reading this, please don't add me to the murder victim statistics.)*  
*(This is simultaneously a hugely unnecessary complexification of a phenomenon that didn't need explaining. Scientific communicators, I'd appreciate if you didn't murder me, but I'd understand it if you do.)*

The code provided here is not intended to respond to any needs other than my own, although I am open to suggestions.  
It is released with an MIT license, with no guarantees of any kind. Fell free to use it in any way you see fit.

*Note: this is very much a work-in-progress. I intend to add better data representations, including the generation of figures, whenever I find the time / motivation.  
Although I do not guarantee that I'll ever finish it, I'd love to know if you found it useful! It might be the source of motivation that I need...*
