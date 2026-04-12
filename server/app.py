from openenv.core.env_server.http_server import create_app

try:
    from ..models import PropertyAction, PropertyObservation
    from .property_environment import PropertyValuationEnvironment
except ImportError:
    from models import PropertyAction, PropertyObservation
    from server.property_environment import PropertyValuationEnvironment

app = create_app(
    PropertyValuationEnvironment,
    PropertyAction,
    PropertyObservation,
    env_name="property-valuation-agent",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()