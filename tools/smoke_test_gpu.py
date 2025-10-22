import os, sys, traceback
import torch

# ensure repository root is on sys.path so `import src` works when running this script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print('Repository root added to sys.path:', ROOT)
print('CUDA available:', torch.cuda.is_available(), 'device count =', torch.cuda.device_count())

errors = []

# Classification smoke: small batch forward/backward with AMP
try:
    from src.classification.model import BaybayinClassifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaybayinClassifier(num_classes=5, weights=None).to(device)
    model.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # prefer torch.amp API
    scaler = torch.amp.GradScaler('cuda') if device.type=='cuda' else None
    x = torch.randn((4,3,224,224), device=device)
    y = torch.randint(0,5,(4,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print('Classification smoke test OK. Loss:', loss.item())
except Exception as e:
    print('Classification smoke test FAILED')
    traceback.print_exc()
    errors.append(('classification', str(e)))

# Detection smoke: build a tiny Faster R-CNN and run a forward pass
try:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    # replace head for 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.to(device)
    model.eval()
    # create a single random image tensor
    img = torch.randn((3,480,640), device=device)
    # forward with no grad to test model inference
    with torch.no_grad():
        outputs = model([img])
    print('Detection smoke test OK. Outputs keys:', list(outputs[0].keys()))
except Exception as e:
    print('Detection smoke test FAILED')
    traceback.print_exc()
    errors.append(('detection', str(e)))

if errors:
    print('SMOKE TEST: FAILURES:', errors)
    sys.exit(2)
else:
    print('SMOKE TEST: ALL OK')
    sys.exit(0)
