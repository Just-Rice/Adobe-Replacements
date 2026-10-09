# EventCenterPlugin

(WIP) Subscribe to events. 

I tried Redis PubSub first, but the plugin I found was buggy. I'm trying WebSockets now.

## Installation

1. Install python and dependencies

```
# Windows works just as Linux,
python -m venv python

# Except for activating the venv
.\python\Scripts\Activate.ps1

# Then it's easy again,
pip install -r requirements.txt
```

2. Install the [BlueprintWebSocket](https://www.unrealengine.com/marketplace/en-US/product/blueprintwebsocket) 
plugin (paid/commercial) from the Epic/Unreal Marketplace. It's not expensive and has great documentation.

3. You need to adjust the main project's build dependencies (`ProjectName.Build.cs`) to include the following,

```
		PublicDependencyModuleNames.AddRange(new string[] { 
			// Preset dependencies
			"Core", "CoreUObject", "Engine", "InputCore", 

			"...",
			
			// EventCenter
			"BlueprintWebSocket",
			"EventCenterPlugin",
		});
```

Unfocused Audio needs to be set in Config/DefaultEngine.ini, otherwise the game won't play audio when unfocused,

```
[Audio]
UnfocusedVolumeMultiplier=1.0
```

## Usage

It's expected to wire events up in the `Level Blueprint`.

## Python testing

I stand up a websocket+pubsub system like so:

```
PS C:\Users\echel\OneDrive\Documents\Unreal Projects\RedisTestProject\Plugins\EventCenterPlugin> .\python\Scripts\Activate.ps1
```

## Notes

A few collected notes, since I'm still new to Unreal engine and I'm cleaning up my debug code.

To log to the game screen,

```
GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "Message goes here");
```

Time based handling in Tick(),

```
void AEventSourceActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	float time = GetGameTimeSinceCreation();

	if (time - timerLastTime > 15.0) {
		timerLastTime = time;
		// Do the thing
	}
}
```

