from glob import glob
import gzip
import os
import pickle
import random
import time

import numpy as np
import cv2

DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/home/mcsmash/dev/data'
PLAY_DIR = f'{DATA_DIR}/game_playing/play_data/'
PNG_TMPL = '{DATA_DIR}/game_playing/play_data/{game}/frames/{play_number}-{frame_number}.png'
PICKLE_DIR = './db/SuperMarioBros-Nes/pickle'

GAMES = {
    'NES': ['1942-Nes', '1943-Nes', '8Eyes-Nes', 'AbadoxTheDeadlyInnerWar-Nes', 'AddamsFamily-Nes', 'AddamsFamilyPugsleysScavengerHunt-Nes', 'AdventureIsland3-Nes', 'AdventureIslandII-Nes', 'AdventuresOfBayouBilly-Nes',
        'AdventuresOfDinoRiki-Nes', 'AdventuresOfRockyAndBullwinkleAndFriends-Nes', 'Airwolf-Nes', 'AlfredChicken-Nes', 'Alien3-Nes', 'AlphaMission-Nes', 'Amagon-Nes', 'Argus-Nes', 'Arkanoid-Nes', 'ArkistasRing-Nes',
        'Armadillo-Nes', 'AstroRoboSasa-Nes', 'Astyanax-Nes', 'Athena-Nes', 'AtlantisNoNazo-Nes', 'AttackAnimalGakuen-Nes', 'AttackOfTheKillerTomatoes-Nes', 'BadDudes-Nes', 'BadStreetBrawler-Nes', 'BalloonFight-Nes',
        'Baltron-Nes', 'BananaPrince-Nes', 'Barbie-Nes', 'BatmanReturns-Nes', 'Battletoads-Nes', 'BinaryLand-Nes', 'BioMiracleBokutteUpa-Nes', 'BioSenshiDanIncreaserTonoTatakai-Nes', 'BirdWeek-Nes', 'BoobyKids-Nes',
        'BoulderDash-Nes', 'BramStokersDracula-Nes', 'BreakThru-Nes', 'BubbleBobble-Nes', 'BubbleBobblePart2-Nes', 'BuckyOHare-Nes', 'BugsBunnyBirthdayBlowout-Nes', 'BuraiFighter-Nes', 'BWings-Nes', 'CaptainAmericaAndTheAvengers-Nes',
        'CaptainPlanetAndThePlaneteers-Nes', 'CaptainSilver-Nes', 'Castelian-Nes', 'CastlevaniaIIIDraculasCurse-Nes', 'Castlevania-Nes', 'CatNindenTeyandee-Nes', 'ChacknPop-Nes', 'Challenger-Nes', 'Choplifter-Nes', 'ChouFuyuuYousaiExedExes-Nes',
        'ChoujikuuYousaiMacross-Nes', 'ChubbyCherub-Nes', 'CircusCaper-Nes', 'CircusCharlie-Nes', 'CityConnection-Nes', 'Cliffhanger-Nes', 'CluCluLand-Nes', 'CobraTriangle-Nes', 'CodeNameViper-Nes', 'Conan-Nes',
        'ConquestOfTheCrystalPalace-Nes', 'ContraForce-Nes', 'CosmicEpsilon-Nes', 'CrossFire-Nes', 'Darkman-Nes', 'DashGalaxyInTheAlienAsylum-Nes', 'DefenderII-Nes', 'DigDugIITroubleInParadise-Nes', 'DiggerTheLegendOfTheLostCity-Nes', 'DirtyHarry-Nes',
        'DonDokoDon-Nes', 'DonkeyKong3-Nes', 'DonkeyKongJr-Nes', 'DonkeyKong-Nes', 'DoubleDragonIITheRevenge-Nes', 'DoubleDragon-Nes', 'DragonPower-Nes', 'DragonSpiritTheNewLegend-Nes', 'ElevatorAction-Nes', 'Exerion-Nes',
        'FantasyZoneIIOpaOpaNoNamida-Nes', 'FelixTheCat-Nes', 'FistOfTheNorthStar-Nes', 'FlintstonesTheRescueOfDinoAndHoppy-Nes', 'FlyingDragonTheSecretScroll-Nes', 'FlyingHero-Nes', 'FormationZ-Nes', 'FoxsPeterPanAndThePiratesTheRevengeOfCaptainHook-Nes', 'FrontLine-Nes', 'GalagaDemonsOfDeath-Nes',
        'Geimos-Nes', 'GhostsnGoblins-Nes', 'GhoulSchool-Nes', 'GIJoeARealAmericanHero-Nes', 'GIJoeTheAtlantisFactor-Nes', 'Gimmick-Nes', 'GradiusII-Nes', 'Gradius-Nes', 'GreatTank-Nes', 'Gremlins2TheNewBatch-Nes',
        'GuardianLegend-Nes', 'GuerrillaWar-Nes', 'GunNac-Nes', 'Gyrodine-Nes', 'Gyruss-Nes', 'HammerinHarry-Nes', 'HeavyBarrel-Nes', 'HelloKittyWorld-Nes', 'HomeAlone2LostInNewYork-Nes', 'HuntForRedOctober-Nes',
        'IceClimber-Nes', 'IkariIIITheRescue-Nes', 'Ikari-Nes', 'Ikki-Nes', 'IndianaJonesAndTheTempleOfDoom-Nes', 'InsectorX-Nes', 'IronSwordWizardsAndWarriorsII-Nes', 'IsolatedWarrior-Nes', 'Jackal-Nes', 'JackieChansActionKungFu-Nes',
        'JajamaruNoDaibouken-Nes', 'JamesBondJr-Nes', 'Jaws-Nes', 'JetsonsCogswellsCaper-Nes', 'JoeAndMac-Nes', 'JourneyToSilius-Nes', 'Joust-Nes', 'JungleBook-Nes', 'KabukiQuantumFighter-Nes', 'KaiketsuYanchaMaru2KarakuriLand-Nes',
        'KaiketsuYanchaMaru3TaiketsuZouringen-Nes', 'KamenNoNinjaAkakage-Nes', 'KanshakudamaNageKantarouNoToukaidouGojuusanTsugi-Nes', 'KeroppiToKeroriinuNoSplashBomb-Nes', 'KidIcarus-Nes', 'KidKlownInNightMayorWorld-Nes', 'KidNikiRadicalNinja-Nes', 'KirbysAdventure-Nes', 'KungFuHeroes-Nes', 'KungFu-Nes',
        'LastActionHero-Nes', 'LastStarfighter-Nes', 'LegendaryWings-Nes', 'LegendOfKage-Nes', 'LegendOfPrinceValiant-Nes', 'LifeForce-Nes', 'LittleMermaid-Nes', 'LowGManTheLowGravityMan-Nes', 'Magmax-Nes', 'MappyLand-Nes',
        'MarioBros-Nes', 'MCKids-Nes', 'MechanizedAttack-Nes', 'MegaMan2-Nes', 'MegaMan-Nes', 'MendelPalace-Nes', 'MetalStorm-Nes', 'MickeyMousecapade-Nes', 'MightyBombJack-Nes', 'MightyFinalFight-Nes',
        'Millipede-Nes', 'MitsumeGaTooru-Nes', 'MoeroTwinBeeCinnamonHakaseOSukue-Nes', 'MonsterInMyPocket-Nes', 'MonsterParty-Nes', 'MsPacMan-Nes', 'MutantVirusCrisisInAComputerWorld-Nes', 'MysteryQuest-Nes', 'NARC-Nes', 'NinjaCrusaders-Nes',
        'NinjaGaidenIIITheAncientShipOfDoom-Nes', 'NinjaGaidenIITheDarkSwordOfChaos-Nes', 'NinjaGaiden-Nes', 'NinjaKid-Nes', 'NoahsArk-Nes', 'OperationWolf-Nes', 'OverHorizon-Nes', 'PacManNamco-Nes', 'PanicRestaurant-Nes', 'Paperboy-Nes',
        'Parodius-Nes', 'PenguinKunWars-Nes', 'PizzaPop-Nes', 'Pooyan-Nes', 'Popeye-Nes', 'POWPrisonersOfWar-Nes', 'Punisher-Nes', 'PussNBootsPerosGreatAdventure-Nes', 'QBert-Nes', 'Quarth-Nes',
        'RainbowIslands-Nes', 'Rampage-Nes', 'Renegade-Nes', 'RoboccoWars-Nes', 'RoboCop2-Nes', 'RoboCop3-Nes', 'RoboWarrior-Nes', 'RockinKats-Nes', 'Rollerball-Nes', 'Rollergames-Nes',
        'RushnAttack-Nes', 'Sansuu5And6NenKeisanGame-Nes', 'SCATSpecialCyberneticAttackTeam-Nes', 'SDHeroSoukessenTaoseAkuNoGundan-Nes', 'SectionZ-Nes', 'Seicross-Nes', 'SeikimaIIAkumaNoGyakushuu-Nes', 'Shatterhand-Nes', 'SilverSurfer-Nes', 'SimpsonsBartmanMeetsRadioactiveMan-Nes',
        'SimpsonsBartVsTheSpaceMutants-Nes', 'SimpsonsBartVsTheWorld-Nes', 'SkyDestroyer-Nes', 'SkyKid-Nes', 'SkyShark-Nes', 'SmashTV-Nes', 'Smurfs-Nes', 'SnakeRattleNRoll-Nes', 'SnowBrothers-Nes', 'SonSon-Nes',
        'SpaceHarrier-Nes', 'SpaceInvaders-Nes', 'SpartanX2-Nes', 'Spelunker-Nes', 'SpyHunter-Nes', 'Sqoon-Nes', 'StarForce-Nes', 'StarshipHector-Nes', 'StarSoldier-Nes', 'StarWars-Nes',
        'Stinger-Nes', 'SuperArabian-Nes', 'SuperC-Nes', 'SuperMarioBros3-Nes', 'SuperMarioBros-Nes', 'SuperPitfall-Nes', 'SuperStarForce-Nes', 'SuperXeviousGumpNoNazo-Nes', 'SwampThing-Nes', 'TaiyouNoYuushaFighbird-Nes',
        'TakahashiMeijinNoBugutteHoney-Nes', 'TargetRenegade-Nes', 'TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes', 'TeenageMutantNinjaTurtlesIITheArcadeGame-Nes', 'TeenageMutantNinjaTurtles-Nes', 'TeenageMutantNinjaTurtlesTournamentFighters-Nes', 'Terminator2JudgmentDay-Nes', 'TerraCresta-Nes', 'TetrastarTheFighter-Nes', 'TetsuwanAtom-Nes',
        'Thexder-Nes', 'ThunderAndLightning-Nes', 'Thundercade-Nes', 'TigerHeli-Nes', 'TimeZone-Nes', 'TinyToonAdventures-Nes', 'Toki-Nes', 'TotallyRad-Nes', 'TotalRecall-Nes', 'ToxicCrusaders-Nes',
        'TreasureMaster-Nes', 'Trog-Nes', 'Trojan-Nes', 'TrollsInCrazyland-Nes', 'TwinBee3PokoPokoDaimaou-Nes', 'TwinBee-Nes', 'TwinCobra-Nes', 'TwinEagle-Nes', 'Untouchables-Nes', 'UrbanChampion-Nes',
        'UruseiYatsuraLumNoWeddingBell-Nes', 'ViceProjectDoom-Nes', 'VolguardII-Nes', 'Warpman-Nes', 'WaynesWorld-Nes', 'Widget-Nes', 'WizardsAndWarriors-Nes', 'WrathOfTheBlackManta-Nes', 'WreckingCrew-Nes', 'Xenophobe-Nes',
        'XeviousTheAvenger-Nes', 'Xexyz-Nes', 'YoukaiClub-Nes', 'YoukaiDouchuuki-Nes', 'YoungIndianaJonesChronicles-Nes', 'Zanac-Nes', ],
    'SNES':[ 'AcceleBrid-Snes', 'ActionPachio-Snes', 'ActRaiser2-Snes', 'AddamsFamilyPugsleysScavengerHunt-Snes', 'AddamsFamily-Snes', 'AdventuresOfDrFranken-Snes', 'AdventuresOfKidKleets-Snes', 'AdventuresOfMightyMax-Snes', 'AdventuresOfRockyAndBullwinkleAndFriends-Snes',
        'AdventuresOfYogiBear-Snes', 'AeroFighters-Snes', 'AeroTheAcroBat2-Snes', 'AeroTheAcroBat-Snes', 'AirCavalry-Snes', 'AlfredChicken-Snes', 'AlienVsPredator-Snes', 'ArcherMacleansSuperDropzone-Snes', 'ArdyLightfoot-Snes', 'ArtOfFighting-Snes',
        'Asterix-Snes', 'Axelay-Snes', 'BatmanReturns-Snes', 'BattleMasterKyuukyokuNoSenshiTachi-Snes', 'BattletoadsDoubleDragon-Snes', 'BattletoadsInBattlemaniacs-Snes', 'BattleZequeDen-Snes', 'BebesKids-Snes', 'BioMetal-Snes', 'BishoujoSenshiSailorMoonR-Snes',
        'BlaZeonTheBioCyborgChallenge-Snes', 'BlockKuzushi-Snes', 'BOB-Snes', 'BoogermanAPickAndFlickAdventure-Snes', 'BramStokersDracula-Snes', 'BrawlBrothers-Snes', 'BronkieTheBronchiasaurus-Snes', 'BubsyII-Snes', 'BubsyInClawsEncountersOfTheFurredKind-Snes', 'CacomaKnightInBizyland-Snes',
        'Cameltry-Snes', 'CannonFodder-Snes', 'CaptainAmericaAndTheAvengers-Snes', 'CaptainCommando-Snes', 'CastlevaniaDraculaX-Snes', 'ChesterCheetahTooCoolToFool-Snes', 'ChesterCheetahWildWildQuest-Snes', 'ChoplifterIIIRescueSurvive-Snes', 'ChoujikuuYousaiMacrossScrambledValkyrie-Snes', 'ChuckRock-Snes',
        'Claymates-Snes', 'Cliffhanger-Snes', 'CongosCaper-Snes', 'CoolSpot-Snes', 'CosmoGangTheVideo-Snes', 'Cybernator-Snes', 'DaffyDuckTheMarvinMissions-Snes', 'DariusForce-Snes', 'DariusTwin-Snes', 'DazeBeforeChristmas-Snes',
        'DennisTheMenace-Snes', 'DimensionForce-Snes', 'DinoCity-Snes', 'DonkeyKongCountry2-Snes', 'DonkeyKongCountry3DixieKongsDoubleTrouble-Snes', 'DonkeyKongCountry-Snes', 'DragonsLair-Snes', 'DragonTheBruceLeeStory-Snes', 'EarthDefenseForce-Snes', 'FamilyDog-Snes',
        'FinalFight2-Snes', 'FinalFight3-Snes', 'FinalFightGuy-Snes', 'FinalFight-Snes', 'FirstSamurai-Snes', 'Flintstones-Snes', 'FlyingHeroBugyuruNoDaibouken-Snes', 'Gods-Snes', 'GokujouParodius-Snes', 'GradiusIII-Snes',
        'GreatCircusMysteryStarringMickeyAndMinnie-Snes', 'HarleysHumongousAdventure-Snes', 'Hook-Snes', 'HuntForRedOctober-Snes', 'Hurricanes-Snes', 'Imperium-Snes', 'Incantation-Snes', 'IncredibleHulk-Snes', 'InspectorGadget-Snes', 'ItchyAndScratchyGame-Snes',
        'IzzysQuestForTheOlympicRings-Snes', 'JellyBoy-Snes', 'JetsonsInvasionOfThePlanetPirates-Snes', 'JoeAndMac2LostInTheTropics-Snes', 'JoeAndMac-Snes', 'JudgeDredd-Snes', 'JungleBook-Snes', 'KaiteTsukutteAsoberuDezaemon-Snes', 'KidKlownInCrazyChase-Snes', 'KingOfDragons-Snes',
        'KingOfTheMonsters2-Snes', 'KnightsOfTheRound-Snes', 'LastActionHero-Snes', 'Legend-Snes', 'LethalWeapon-Snes', 'MagicalQuestStarringMickeyMouse-Snes', 'MagicBoy-Snes', 'MagicSword-Snes', 'Mask-Snes', 'MightyMorphinPowerRangersTheMovie-Snes',
        'MrNutz-Snes', 'OutToLunch-Snes', 'PacInTime-Snes', 'Parodius-Snes', 'PeaceKeepers-Snes', 'Phalanx-Snes', 'PiratesOfDarkWater-Snes', 'PitfallTheMayanAdventure-Snes', 'Plok-Snes', 'PopnTwinBeeRainbowBellAdventures-Snes',
        'PopnTwinBee-Snes', 'PorkyPigsHauntedHoliday-Snes', 'PowerPiggsOfTheDarkAge-Snes', 'PrehistorikMan-Snes', 'PuttySquad-Snes', 'RadicalRex-Snes', 'RaidenDensetsu-Snes', 'Realm-Snes', 'RenAndStimpyShowVeediots-Snes', 'RenderingRangerR2-Snes',
        'RivalTurf-Snes', 'RoadRunnersDeathValleyRally-Snes', 'RoboCop3-Snes', 'RoboCopVersusTheTerminator-Snes', 'RTypeIII-Snes', 'RunSaber-Snes', 'SkuljaggerRevoltOfTheWesticans-Snes', 'SmartBall-Snes', 'Smurfs-Snes', 'SonicBlastManII-Snes',
        'SonicBlastMan-Snes', 'SonicWings-Snes', 'SpaceInvaders-Snes', 'SpaceMegaforce-Snes', 'SpankysQuest-Snes', 'Sparkster-Snes', 'SpeedyGonzalesLosGatosBandidos-Snes', 'SprigganPowered-Snes', 'StoneProtectors-Snes', 'SuperAdventureIsland-Snes',
        'SuperAlfredChicken-Snes', 'SuperBCKid-Snes', 'SuperCastlevaniaIV-Snes', 'SuperDoubleDragon-Snes', 'SuperGhoulsnGhosts-Snes', 'SuperJamesPond-Snes', 'SuperMarioWorld2-Snes', 'SuperMarioWorld-Snes', 'SuperRType-Snes', 'SuperSmashTV-Snes',
        'SuperStarWarsReturnOfTheJedi-Snes', 'SuperStarWars-Snes', 'SuperStarWarsTheEmpireStrikesBack-Snes', 'SuperStrikeGunner-Snes', 'SuperSWIV-Snes', 'SuperTrollIslands-Snes', 'SuperTurrican2-Snes', 'SuperTurrican-Snes', 'SuperValisIV-Snes', 'SuperWidget-Snes',
        'TazMania-Snes', 'TeenageMutantNinjaTurtlesIVTurtlesInTime-Snes', 'TetrisAttack-Snes', 'ThunderSpirits-Snes', 'Tick-Snes', 'TinyToonAdventuresBusterBustsLoose-Snes', 'TomAndJerry-Snes', 'UchuuNoKishiTekkamanBlade-Snes', 'UNSquadron-Snes', 'WereBackADinosaursStory-Snes',
        'Wolfchild-Snes', 'XKaliber2097-Snes', 'ZeroTheKamikazeSquirrel-Snes', 'ZombiesAteMyNeighbors-Snes', 'ZoolNinjaOfTheNthDimension-Snes', ],
}

def neighboring_points(x, y, arr, indirect=True):
    max_x, max_y = arr.shape[:2]
    neighbors = []
    if x > 0 and y > 0 and indirect is True:
        neighbors.append((x-1, y-1))
    if y > 0:
        neighbors.append((x, y-1))
    if x < max_x - 1 and y > 0 and indirect is True:
        neighbors.append((x+1, y-1))

    if x > 0:
        neighbors.append((x-1, y))
    #if x > 0 and np.array_equal(frame[x][y], frame[x][y]):
    #    neighbors.append((x, y))
    if x < max_x - 1:
        neighbors.append((x+1, y))

    if x > 0 and y < max_y - 1 and indirect is True:
        neighbors.append((x-1, y+1))
    if y < max_y - 1:
        neighbors.append((x, y+1))
    if x < max_x - 1 and y < max_y - 1 and indirect is True:
        neighbors.append((x+1, y+1))

    return neighbors

def show_image(img, scale=1.0):
    cv2.imshow('frame', cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)

def get_image_list(game='SuperMarioBros-Nes', play_number=None):
    print('Loading image file list')
    if play_number is None:
        with open('./image_files', 'r') as fp:
            frame_paths = [i.strip() for i in fp.readlines()]
        random.shuffle(frame_paths)
        return frame_paths
    else:
        return glob(f'{PLAY_DIR}/{game}/{play_number}/*')

def get_frame(game, play_number, frame_number):
    return cv2.imread(PNG_TMPL.format(DATA_DIR=DATA_DIR, game=game, play_number=play_number, frame_number=frame_number))
def get_playthrough(game='SuperMarioBros-Nes', play_number=None):
    if play_number is not None:
        d = np.load(f'{PLAY_DIR}/{game}/{play_number}/frames.npz')
        return d['arr_0']
    else:
        raise TypeError

def load_indexed_playthrough(play_number, count=None):
    db_path = get_db_path(play_number)
    if not ensure_dir(db_path):
        frame_paths = []
    else:
        frame_paths = glob(f'{db_path}/*')

    start = time.time()
    if count is None:
        number_of_frames = len(frame_paths)
    else:
        number_of_frames = count

    for i in range(number_of_frames):
        print(f'loading: {i}')
        with gzip.GzipFile(f'{PICKLE_DIR}/{play_number}/{i}.pickle', 'rb') as fp:
            yield pickle.load(fp)

def get_db_path(play_number, backend='pickle'):
    if backend == 'pickle':
        return f'{PICKLE_DIR}/{play_number}'

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return False
    return True

