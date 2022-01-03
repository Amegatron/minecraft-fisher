# Minecraft fisher

A bot for fishing in Minecraft using just game visuals

## Intro

First, demo video of it working: https://youtu.be/FXoB0K70lb4

**DISCLAIMER:** This bot was initially made as just a practice in the field of Computer Vision (CV) and Deep Learning (DL)
and at its current state is mainly a proof-of-concept work, showing, that such botting is possible.
This means that though it works in general in most "usual" scenarios (like showed in the video above), including night and even
during rain, it can still fail in some non-standard ones. But this is mostly a matter of collecting
a vaster training dataset (see further). This also means that the bot in its current state is (1) not optimized at all yet,
so I believe users on low- or maybe even mid-end computers may experience performance issues; and (2) it is not much user-friendly.
But still, it is actually working and may prove useful to somebody.

## Simple usage "as is"

First, I assume you have downloaded or cloned this repo and have python and all the dependecies installed.
Sorry for not listing the details of this process yet, hope you'll manage it yourself.

Then, if you just want to use it as is and don't want to train it yourself, you'd need to
download a pretrained state of the bot (`fisherman_state.pt`) from [here](https://drive.google.com/file/d/1Lp0SOxq3tlVs1MKDXxcKZQHNqixCb-E4/view?usp=sharing)
and place it just in the folder where you put these sources.

Next make sure Minecraft is already launched and you're already in the game. Next, before starting the fishing process itself
you need to take the Fishing Rod in your hand and manually approach the position appropriate for fishing. This literally means
that your player must be in position, where right-clicking will cast a rod.

Next, launching the bot itself. From command line and assuming you're already inside the folder with this bot
(and you're on Windows) you need to execute this command:

```shell
python.exe bot.py "Minecraft 1.18"
```

Where "Minecraft 1.18" - is part of Minecraft's window title, so it can be unambiguously identified.
For example, if you're playing version 1.18.1 and you're currently in single-player, the title of 
the window most probable is "Minecraft 1.18.1 - Singleplayer", but you can still pass "Minecraft 1.18"
to the bot if that's the only such window.

After the bot has started, it will first load the pretrained model you placed earlier, and soon will
ask you to press Enter to start actual botting. After that, a countdown will start so you have time to switch
back to the game itself and overall make sure the game window is in the foreground. This is needed because in
vanilla Minecraft client, the game starts showing the menu as soon as you Alt-Tab so anything else, in fact hiding the
picture of the game and stopping to receive any controls for a character. But you can still make your own experiments
if the bot works with "no-alt-tab" mods. Though I haven't tested that, it should theoretically work.

And that's it. In case you need to temporary alt-tab anyway, the bot may get stuck. In this case you can
just manually "fix" the state by just grabbing the bobber back if it was in water, and the bot shall continue.

## Training the bot on your own

Instructions will be later. But for those willing to figure it out themselves, here is a training dataset
which I collected and used to train the bot: https://drive.google.com/file/d/1Br7h8YFxq1spPJBLY5YvmKge51KCq2QV/view?usp=sharing